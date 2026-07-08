"""Implementación del backend de procesamiento para GPU NVIDIA (CUDA)."""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable

import numpy as np

from core.backend.base import (
    GPUBackend,
    CUDA_BACKEND_NAME,
    MAX_PIXEL_VALUE,
    NO_BATCH_TIME,
    VALID_OPERATIONS,
    BLUR_RADIUS,
    GAUSSIAN_BLUR_WEIGHTS_1D,
    HISTOGRAM_BINS,
    compute_equalize_lut,
    _gpu_semaphore,
    OP_GRAYSCALE,
    OP_EDGES,
    OP_BLUR,
    OP_EQUALIZE,
    GRAYSCALE_R,
    GRAYSCALE_G,
    GRAYSCALE_B,
    EDGE_MARGIN,
    SOBEL_WEIGHT,
    WARMUP_IMAGE_SIZE,
    GPU_THREADS_PER_BLOCK,
    RGB_CHANNELS,
)
from core.backend.cpu import CPUBackend

_LOGGER = logging.getLogger(__name__)
_COMPUTE_TLS = threading.local()

if sys.platform == 'win32':
    try:
        cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
        if cuda_path and os.path.isdir(cuda_path):
            bin_x64 = os.path.join(cuda_path, 'bin', 'x64')
            nvvm_bin_x64 = os.path.join(cuda_path, 'nvvm', 'bin', 'x64')
            if hasattr(os, 'add_dll_directory'):
                if os.path.isdir(bin_x64):
                    os.add_dll_directory(bin_x64)
                if os.path.isdir(nvvm_bin_x64):
                    os.add_dll_directory(nvvm_bin_x64)
            try:
                from numba.cuda import cuda_paths
                paths = cuda_paths.get_cuda_paths()
                if os.path.isdir(bin_x64) and paths['cudalib_dir'].info != bin_x64:
                    paths['cudalib_dir'] = cuda_paths._env_path_tuple(
                        paths['cudalib_dir'].by, bin_x64
                    )
                if os.path.isdir(nvvm_bin_x64) and not paths['nvvm'].info:
                    import glob
                    nvvm_dlls = glob.glob(os.path.join(nvvm_bin_x64, 'nvvm*.dll'))
                    if nvvm_dlls:
                        paths['nvvm'] = cuda_paths._env_path_tuple(
                            paths['nvvm'].by, nvvm_dlls[0]
                        )
            except Exception:
                pass
    except Exception:
        pass

try:
    from numba import cuda
    _CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    _CUDA_AVAILABLE = False


CUDA_GRID_DIM: int = 2
CUDA_GRID_DIM_BATCH: int = 3

if _CUDA_AVAILABLE:

    @cuda.jit
    def grayscale_kernel(image, output, rows, cols):
        x, y = cuda.grid(CUDA_GRID_DIM)
        if x < rows and y < cols:
            r = image[x, y, 0]
            g = image[x, y, 1]
            b = image[x, y, 2]
            output[x, y] = GRAYSCALE_R * r + GRAYSCALE_G * g + GRAYSCALE_B * b

    @cuda.jit
    def edges_kernel(image, output, rows, cols):
        x, y = cuda.grid(CUDA_GRID_DIM)
        if EDGE_MARGIN <= x < rows - EDGE_MARGIN and EDGE_MARGIN <= y < cols - EDGE_MARGIN:
            gx = (
                -float(image[x-EDGE_MARGIN, y-EDGE_MARGIN]) + float(image[x-EDGE_MARGIN, y+EDGE_MARGIN])
                - SOBEL_WEIGHT*float(image[x, y-EDGE_MARGIN]) + SOBEL_WEIGHT*float(image[x, y+EDGE_MARGIN])
                - float(image[x+EDGE_MARGIN, y-EDGE_MARGIN]) + float(image[x+EDGE_MARGIN, y+EDGE_MARGIN])
            )
            gy = (
                -float(image[x-EDGE_MARGIN, y-EDGE_MARGIN]) - SOBEL_WEIGHT*float(image[x-EDGE_MARGIN, y]) - float(image[x-EDGE_MARGIN, y+EDGE_MARGIN])
                + float(image[x+EDGE_MARGIN, y-EDGE_MARGIN]) + SOBEL_WEIGHT*float(image[x+EDGE_MARGIN, y]) + float(image[x+EDGE_MARGIN, y+EDGE_MARGIN])
            )
            magnitude = math.sqrt(gx*gx + gy*gy)
            output[x, y] = min(magnitude, MAX_PIXEL_VALUE)

    @cuda.jit
    def blur_kernel(image, output, mask, radius, rows, cols, channels):
        x, y = cuda.grid(CUDA_GRID_DIM)
        if x < rows and y < cols:
            for c in range(channels):
                acc = 0.0
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        px = min(max(x + i, 0), rows - 1)
                        py = min(max(y + j, 0), cols - 1)
                        acc += mask[i + radius, j + radius] * image[px, py, c]
                output[x, y, c] = acc

    @cuda.jit
    def histogram_kernel(gray, histogram, rows, cols):
        x, y = cuda.grid(CUDA_GRID_DIM)
        if x < rows and y < cols:
            cuda.atomic.add(histogram, gray[x, y], 1)

    @cuda.jit
    def map_lut_kernel(gray, lut, output, rows, cols):
        x, y = cuda.grid(CUDA_GRID_DIM)
        if x < rows and y < cols:
            output[x, y] = lut[gray[x, y]]

    @cuda.jit
    def grayscale_batch_kernel(images, output, n_imgs, rows, cols):
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)
        if n < n_imgs and x < rows and y < cols:
            r = images[n, x, y, 0]
            g = images[n, x, y, 1]
            b = images[n, x, y, 2]
            output[n, x, y] = GRAYSCALE_R * r + GRAYSCALE_G * g + GRAYSCALE_B * b

    @cuda.jit
    def edges_batch_kernel(grays, output, n_imgs, rows, cols):
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)
        if n < n_imgs and EDGE_MARGIN <= x < rows - EDGE_MARGIN and EDGE_MARGIN <= y < cols - EDGE_MARGIN:
            gx = (
                -float(grays[n, x-1, y-1]) + float(grays[n, x-1, y+1])
                - SOBEL_WEIGHT*float(grays[n, x, y-1]) + SOBEL_WEIGHT*float(grays[n, x, y+1])
                - float(grays[n, x+1, y-1]) + float(grays[n, x+1, y+1])
            )
            gy = (
                -float(grays[n, x-1, y-1]) - SOBEL_WEIGHT*float(grays[n, x-1, y]) - float(grays[n, x-1, y+1])
                + float(grays[n, x+1, y-1]) + SOBEL_WEIGHT*float(grays[n, x+1, y]) + float(grays[n, x+1, y+1])
            )
            output[n, x, y] = min(math.sqrt(gx*gx + gy*gy), MAX_PIXEL_VALUE)

    @cuda.jit
    def blur_batch_kernel(images, output, mask, radius, n_imgs, rows, cols, channels):
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)
        if n < n_imgs and x < rows and y < cols:
            for c in range(channels):
                acc = 0.0
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        px = min(max(x + i, 0), rows - 1)
                        py = min(max(y + j, 0), cols - 1)
                        acc += mask[i + radius, j + radius] * images[n, px, py, c]
                output[n, x, y, c] = acc

    @cuda.jit
    def histogram_batch_kernel(grays, histograms, n_imgs, rows, cols):
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)
        if n < n_imgs and x < rows and y < cols:
            cuda.atomic.add(histograms, (n, int(grays[n, x, y])), 1)

    @cuda.jit
    def map_lut_batch_kernel(grays, luts, output, n_imgs, rows, cols):
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)
        if n < n_imgs and x < rows and y < cols:
            output[n, x, y] = luts[n, grays[n, x, y]]

    try:
        _GPU_OOM_ERRORS: tuple[type[Exception], ...] = (
            MemoryError, cuda.cudadrv.driver.CudaAPIError)
    except AttributeError:
        _GPU_OOM_ERRORS = (MemoryError,)
else:
    _GPU_OOM_ERRORS = (MemoryError,)

def _gaussian_mask() -> np.ndarray:
    weights = np.array(GAUSSIAN_BLUR_WEIGHTS_1D, dtype=np.float32)
    return np.outer(weights, weights)

class CUDABackend(GPUBackend):
    """Procesa imágenes en GPU NVIDIA mediante kernels CUDA (Numba)."""

    def __init__(self, on_fallback: Callable[[], None] | None = None):
        self.backend_name = CUDA_BACKEND_NAME
        self._on_fallback = on_fallback
        self._cpu_fallback = CPUBackend()
        try:
            device = cuda.get_current_device()
            self.device_info = str(device)
        except AttributeError:
            self.device_info = "CUDA Simulator"

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        if operation not in VALID_OPERATIONS:
            raise ValueError(f'Unknown operation: {operation!r}')
        with _gpu_semaphore:
            try:
                start = time.perf_counter()
                if operation == OP_GRAYSCALE:
                    result = self._run_grayscale(image)
                elif operation == OP_EDGES:
                    result = self._run_edges(image)
                elif operation == OP_BLUR:
                    result = self._run_blur(image)
                elif operation == OP_EQUALIZE:
                    result = self._run_equalize(image)
                _COMPUTE_TLS.last_ms = (
                    (time.perf_counter() - start) * 1000.0)
                return result
            except _GPU_OOM_ERRORS as exc:
                self._handle_oom(image, exc)
                return self._cpu_fallback.process(image, operation)

    def warmup(self, operation: str) -> None:
        if not _CUDA_AVAILABLE:
            return
        _shape = (WARMUP_IMAGE_SIZE, WARMUP_IMAGE_SIZE, RGB_CHANNELS)
        dummy = np.zeros(_shape, dtype=np.uint8)
        self.process(dummy, operation)
        self.process_batch([dummy, dummy], operation)

    def process_batch(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> tuple[list[np.ndarray], float | None]:
        with _gpu_semaphore:
            try:
                start = time.perf_counter()
                if operation == OP_GRAYSCALE:
                    results = self._batch_grayscale(images)
                elif operation == OP_EDGES:
                    results = self._batch_edges(images)
                elif operation == OP_BLUR:
                    results = self._batch_blur(images)
                elif operation == OP_EQUALIZE:
                    results = self._batch_equalize(images)
                ms = (time.perf_counter() - start) * 1000.0
                return results, ms / len(images)
            except _GPU_OOM_ERRORS as exc:
                self._handle_oom(images[0], exc)
                fallback = [self._cpu_fallback.process(img, operation)
                            for img in images]
                return fallback, NO_BATCH_TIME

    def _run_grayscale(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        d_image = cuda.to_device(image)
        d_output = cuda.device_array((rows, cols), dtype=np.uint8)
        blocks, threads = self._get_grid_dims(image.shape)
        grayscale_kernel[blocks, threads](d_image, d_output, rows, cols)
        cuda.synchronize()
        return d_output.copy_to_host()

    def _run_edges(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        d_image = cuda.to_device(image)
        d_gray = cuda.device_array((rows, cols), dtype=np.uint8)
        d_output = cuda.device_array((rows, cols), dtype=np.uint8)
        blocks, threads = self._get_grid_dims(image.shape)
        
        grayscale_kernel[blocks, threads](d_image, d_gray, rows, cols)
        edges_kernel[blocks, threads](d_gray, d_output, rows, cols)
        cuda.synchronize()
        return d_output.copy_to_host()

    def _run_blur(self, image: np.ndarray) -> np.ndarray:
        rows, cols, channels = image.shape
        d_image = cuda.to_device(image)
        d_output = cuda.device_array(image.shape, dtype=np.uint8)
        d_mask = cuda.to_device(_gaussian_mask())
        blocks, threads = self._get_grid_dims(image.shape)
        blur_kernel[blocks, threads](d_image, d_output, d_mask, BLUR_RADIUS, rows, cols, channels)
        cuda.synchronize()
        return d_output.copy_to_host()

    def _run_equalize(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        d_image = cuda.to_device(image)
        d_gray = cuda.device_array((rows, cols), dtype=np.uint8)
        blocks, threads = self._get_grid_dims(image.shape)
        
        grayscale_kernel[blocks, threads](d_image, d_gray, rows, cols)
        
        d_hist = cuda.to_device(np.zeros(HISTOGRAM_BINS, dtype=np.int32))
        histogram_kernel[blocks, threads](d_gray, d_hist, rows, cols)
        cuda.synchronize()
        
        histogram = d_hist.copy_to_host()
        lut = compute_equalize_lut(histogram)
        
        d_lut = cuda.to_device(lut)
        d_output = cuda.device_array((rows, cols), dtype=np.uint8)
        map_lut_kernel[blocks, threads](d_gray, d_lut, d_output, rows, cols)
        cuda.synchronize()
        
        return d_output.copy_to_host()

    def _batch_grayscale(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        rows, cols = images[0].shape[:2]
        stacked = np.ascontiguousarray(np.stack(images))
        d_in = cuda.to_device(stacked)
        d_out = cuda.device_array((n, rows, cols), dtype=np.uint8)
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        grayscale_batch_kernel[blocks, threads](d_in, d_out, n, rows, cols)
        cuda.synchronize()
        host = d_out.copy_to_host()
        return [host[i] for i in range(n)]

    def _batch_edges(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        rows, cols = images[0].shape[:2]
        stacked = np.ascontiguousarray(np.stack(images))
        d_in = cuda.to_device(stacked)
        d_gray = cuda.device_array((n, rows, cols), dtype=np.uint8)
        d_out = cuda.device_array((n, rows, cols), dtype=np.uint8)
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        grayscale_batch_kernel[blocks, threads](d_in, d_gray, n, rows, cols)
        edges_batch_kernel[blocks, threads](d_gray, d_out, n, rows, cols)
        cuda.synchronize()
        host = d_out.copy_to_host()
        return [host[i] for i in range(n)]

    def _batch_blur(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        rows, cols, channels = images[0].shape
        stacked = np.ascontiguousarray(np.stack(images))
        d_in = cuda.to_device(stacked)
        d_out = cuda.device_array((n, rows, cols, channels), dtype=np.uint8)
        d_mask = cuda.to_device(_gaussian_mask())
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        blur_batch_kernel[blocks, threads](d_in, d_out, d_mask, BLUR_RADIUS, n, rows, cols, channels)
        cuda.synchronize()
        host = d_out.copy_to_host()
        return [host[i] for i in range(n)]

    def _batch_equalize(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        rows, cols = images[0].shape[:2]
        stacked = np.ascontiguousarray(np.stack(images))
        d_in = cuda.to_device(stacked)
        d_gray = cuda.device_array((n, rows, cols), dtype=np.uint8)
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        
        grayscale_batch_kernel[blocks, threads](d_in, d_gray, n, rows, cols)
        
        h_hists = np.zeros((n, HISTOGRAM_BINS), dtype=np.int32)
        d_hists = cuda.to_device(h_hists)
        histogram_batch_kernel[blocks, threads](d_gray, d_hists, n, rows, cols)
        cuda.synchronize()
        
        hists = d_hists.copy_to_host()
        luts = np.stack([compute_equalize_lut(hists[i]) for i in range(n)])
        
        d_luts = cuda.to_device(luts)
        d_out = cuda.device_array((n, rows, cols), dtype=np.uint8)
        map_lut_batch_kernel[blocks, threads](d_gray, d_luts, d_out, n, rows, cols)
        cuda.synchronize()
        
        host = d_out.copy_to_host()
        return [host[i] for i in range(n)]

    def _handle_oom(self, image: np.ndarray, exc: Exception) -> None:
        _LOGGER.warning(
            'GPU OOM (%s), fallback a CPU: %r', image.shape, exc)
        if self._on_fallback is not None:
            self._on_fallback()

    def _get_grid_dims(self, shape: tuple) -> tuple[tuple, tuple]:
        threads = (GPU_THREADS_PER_BLOCK, GPU_THREADS_PER_BLOCK)
        blocks = (
            math.ceil(shape[0] / threads[0]),
            math.ceil(shape[1] / threads[1]),
        )
        return blocks, threads

    def _get_batch_grid_dims(
        self, shape: tuple, batch_n: int
    ) -> tuple[tuple, tuple]:
        threads = (GPU_THREADS_PER_BLOCK, GPU_THREADS_PER_BLOCK, 1)
        blocks = (
            math.ceil(shape[0] / GPU_THREADS_PER_BLOCK),
            math.ceil(shape[1] / GPU_THREADS_PER_BLOCK),
            batch_n,
        )
        return blocks, threads
