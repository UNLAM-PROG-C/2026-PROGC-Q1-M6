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
)
from core.backend.cpu import CPUBackend, _to_grayscale

_LOGGER = logging.getLogger(__name__)
_COMPUTE_TLS = threading.local()

# Parche para detectar CUDA Toolkit en Windows x64 (v12/v13) con Numba
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


# CONSTANTES CUDA
WARMUP_IMAGE_SIZE: int = 32
RGB_CHANNELS: int = 3
_MS_PER_SECOND: float = 1000.0

CUDA_THREADS_PER_BLOCK: int = 16
CUDA_GRID_DIM: int = 2
CUDA_GRID_DIM_BATCH: int = 3
EDGE_MARGIN: int = 1
SOBEL_WEIGHT: int = 2

# Formula de conversion a grayscale:
CUDA_GRAYSCALE_RED_WEIGHT: float = 0.299
CUDA_GRAYSCALE_GREEN_WEIGHT: float = 0.587
CUDA_GRAYSCALE_BLUE_WEIGHT: float = 0.114

if _CUDA_AVAILABLE:

    @cuda.jit
    def _grayscale_kernel(image, output):
        """Kernel CUDA de conversión a escala de grises."""
        x, y = cuda.grid(CUDA_GRID_DIM)  # pylint: disable=no-value-for-parameter
        rows, cols = image.shape[0], image.shape[1]
        if x < rows and y < cols:  # pylint: disable=comparison-with-callable
            # Canal 0: Red, 1: Green, 2: Blue
            channel_r, channel_g, channel_b = 0, 1, 2
            r = image[x, y, channel_r]
            g = image[x, y, channel_g]
            b = image[x, y, channel_b]
            output[x, y] = (
                CUDA_GRAYSCALE_RED_WEIGHT * r
                + CUDA_GRAYSCALE_GREEN_WEIGHT * g
                + CUDA_GRAYSCALE_BLUE_WEIGHT * b
            )

    @cuda.jit
    def _edges_kernel(image, output):
        """Kernel CUDA de detección de bordes."""
        # pylint: disable=no-value-for-parameter
        x, y = cuda.grid(CUDA_GRID_DIM)
        if (EDGE_MARGIN <= x < image.shape[0] - EDGE_MARGIN and
                EDGE_MARGIN <= y < image.shape[1] - EDGE_MARGIN):

            # Casteamos a float para evitar overflow/underflow de uint8.
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
    def _blur_kernel(image, output, mask, radius):
        """Kernel CUDA de desenfoque gaussiano por canal con clamp."""
        x, y = cuda.grid(CUDA_GRID_DIM)  # pylint: disable=no-value-for-parameter
        rows, cols, channels = image.shape
        if x < rows and y < cols:  # pylint: disable=comparison-with-callable
            for c in range(channels):
                acc = 0.0
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        px = min(max(x + i, 0), rows - 1)
                        py = min(max(y + j, 0), cols - 1)
                        acc += mask[i + radius, j + radius] * image[px, py, c]
                output[x, y, c] = acc

    @cuda.jit
    def _histogram_kernel(gray, histogram):
        """Kernel CUDA que acumula el histograma con sumas atómicas."""
        x, y = cuda.grid(CUDA_GRID_DIM)  # pylint: disable=no-value-for-parameter
        rows, cols = gray.shape[0], gray.shape[1]
        if x < rows and y < cols:  # pylint: disable=comparison-with-callable
            # pylint: disable-next=too-many-function-args
            cuda.atomic.add(histogram, gray[x, y], 1)

    @cuda.jit
    def _map_kernel(gray, lut, output):
        """Kernel CUDA que aplica la LUT de ecualización por píxel."""
        x, y = cuda.grid(CUDA_GRID_DIM)  # pylint: disable=no-value-for-parameter
        rows, cols = gray.shape[0], gray.shape[1]
        if x < rows and y < cols:  # pylint: disable=comparison-with-callable
            output[x, y] = lut[gray[x, y]]

    @cuda.jit
    def _grayscale_batch_kernel(images, output):
        """Kernel CUDA batcheado de conversión a escala de grises."""
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)  # pylint: disable=no-value-for-parameter
        n_imgs, rows, cols = output.shape[0], output.shape[1], output.shape[2]
        if n < n_imgs and x < rows and y < cols:  # pylint: disable=comparison-with-callable
            channel_r, channel_g, channel_b = 0, 1, 2
            r = images[n, x, y, channel_r]
            g = images[n, x, y, channel_g]
            b = images[n, x, y, channel_b]
            output[n, x, y] = (
                CUDA_GRAYSCALE_RED_WEIGHT * r
                + CUDA_GRAYSCALE_GREEN_WEIGHT * g
                + CUDA_GRAYSCALE_BLUE_WEIGHT * b
            )

    @cuda.jit
    def _edges_batch_kernel(grays, output):
        """Kernel CUDA batcheado de detección de bordes (Sobel)."""
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)  # pylint: disable=no-value-for-parameter
        n_imgs, rows, cols = output.shape[0], output.shape[1], output.shape[2]
        if (n < n_imgs and EDGE_MARGIN <= x < rows - EDGE_MARGIN
                and EDGE_MARGIN <= y < cols - EDGE_MARGIN):
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
    def _blur_batch_kernel(images, output, mask, radius):
        """Kernel CUDA batcheado de desenfoque gaussiano por canal."""
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)  # pylint: disable=no-value-for-parameter
        n_imgs, rows, cols, channels = output.shape
        if n < n_imgs and x < rows and y < cols:  # pylint: disable=comparison-with-callable
            for c in range(channels):
                acc = 0.0
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        px = min(max(x + i, 0), rows - 1)
                        py = min(max(y + j, 0), cols - 1)
                        acc += mask[i + radius, j + radius] * images[n, px, py, c]
                output[n, x, y, c] = acc

    @cuda.jit
    def _histogram_batch_kernel(grays, histograms):
        """Kernel CUDA batcheado que acumula histogramas con sumas atómicas."""
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)  # pylint: disable=no-value-for-parameter
        n_imgs, rows, cols = grays.shape[0], grays.shape[1], grays.shape[2]
        if n < n_imgs and x < rows and y < cols:  # pylint: disable=comparison-with-callable
            # pylint: disable-next=too-many-function-args
            cuda.atomic.add(histograms, (n, int(grays[n, x, y])), 1)

    @cuda.jit
    def _map_batch_kernel(grays, luts, output):
        """Kernel CUDA batcheado que aplica LUTs de ecualización por píxel."""
        x, y, n = cuda.grid(CUDA_GRID_DIM_BATCH)  # pylint: disable=no-value-for-parameter
        n_imgs, rows, cols = grays.shape[0], grays.shape[1], grays.shape[2]
        if n < n_imgs and x < rows and y < cols:  # pylint: disable=comparison-with-callable
            output[n, x, y] = luts[n, grays[n, x, y]]

    _OPERATIONS_CUDA = {
        'grayscale': _grayscale_kernel,
        'edges': _edges_kernel,
    }
    try:
        _GPU_OOM_ERRORS: tuple[type[Exception], ...] = (
            MemoryError, cuda.cudadrv.driver.CudaAPIError)
    except AttributeError:
        _GPU_OOM_ERRORS = (MemoryError,)
else:
    _OPERATIONS_CUDA = {}
    _GPU_OOM_ERRORS = (MemoryError,)


def _gaussian_mask() -> np.ndarray:
    """Construye la máscara gaussiana 2D normalizada para el desenfoque."""
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
        """Aplica la operación a la imagen usando un kernel CUDA.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformación a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado; cae a CPUBackend ante un OOM de GPU.

        Raises:
            ValueError: Si operation no está en VALID_OPERATIONS.
        """
        if operation not in VALID_OPERATIONS:
            raise ValueError(f'Unknown operation: {operation!r}')
        with _gpu_semaphore:
            try:
                start = time.perf_counter()
                result = self._process_on_gpu(image, operation)
                _COMPUTE_TLS.last_ms = (
                    (time.perf_counter() - start) * _MS_PER_SECOND)
                return result
            except _GPU_OOM_ERRORS as exc:
                self._handle_oom(image, exc)
                return self._cpu_fallback.process(image, operation)

    def warmup(self, operation: str) -> None:
        """Compila kernels CUDA con imagen dummy; descarta el tiempo."""
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
        """Procesa el lote en una sola transferencia PCIe H2D y D2H.

        Args:
            images: Lista de arrays del mismo shape, dtype uint8.
            operation: Transformación a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Tupla (resultados, per_image_ms) con tiempo amortizado por imagen.
        """
        with _gpu_semaphore:
            try:
                start = time.perf_counter()
                results = self._dispatch_batch(images, operation)
                ms = (time.perf_counter() - start) * _MS_PER_SECOND
                return results, ms / len(images)
            except _GPU_OOM_ERRORS as exc:
                self._handle_oom(images[0], exc)
                fallback = [self._cpu_fallback.process(img, operation)
                            for img in images]
                return fallback, NO_BATCH_TIME

    def _dispatch_batch(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> list[np.ndarray]:
        """Enruta el lote al helper correspondiente a la operación."""
        if operation == 'blur':
            return self._batch_blur(images)
        if operation == 'equalize':
            return self._batch_equalize(images)
        return self._batch_simple(images, operation)

    def _batch_simple(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> list[np.ndarray]:
        """Enruta el lote a grayscale o edges en GPU."""
        if operation == 'edges':
            return self._batch_edges(images)
        return self._batch_grayscale(images)

    def _batch_grayscale(
        self, images: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Convierte N imágenes a escala de grises en un solo kernel."""
        n = len(images)
        stacked = np.ascontiguousarray(np.stack(images))
        h, w = images[0].shape[0], images[0].shape[1]
        d_in = cuda.to_device(stacked)
        d_out = cuda.device_array((n, h, w), dtype=np.uint8)
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        # pylint: disable-next=possibly-used-before-assignment
        _grayscale_batch_kernel[blocks, threads](d_in, d_out)
        cuda.synchronize()
        host = d_out.copy_to_host()
        return [host[i] for i in range(n)]

    def _batch_edges(
        self, images: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Aplica Sobel en lote; la conversión a gris ocurre en GPU."""
        n = len(images)
        h, w = images[0].shape[0], images[0].shape[1]
        stacked = np.ascontiguousarray(np.stack(images))
        d_in = cuda.to_device(stacked)
        d_gray = cuda.device_array((n, h, w), dtype=np.uint8)
        d_out = cuda.device_array((n, h, w), dtype=np.uint8)
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        # pylint: disable-next=possibly-used-before-assignment
        _grayscale_batch_kernel[blocks, threads](d_in, d_gray)
        # pylint: disable-next=possibly-used-before-assignment
        _edges_batch_kernel[blocks, threads](d_gray, d_out)
        cuda.synchronize()
        host = d_out.copy_to_host()
        return [host[i] for i in range(n)]

    def _batch_blur(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Aplica desenfoque gaussiano en lote con un solo kernel CUDA."""
        n = len(images)
        stacked = np.ascontiguousarray(np.stack(images))
        n_imgs, h, w, c = stacked.shape
        d_in = cuda.to_device(stacked)
        d_out = cuda.device_array((n_imgs, h, w, c), dtype=np.uint8)
        d_mask = cuda.to_device(_gaussian_mask())
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        # pylint: disable-next=possibly-used-before-assignment
        _blur_batch_kernel[blocks, threads](d_in, d_out, d_mask, BLUR_RADIUS)
        cuda.synchronize()
        host = d_out.copy_to_host()
        return [host[i] for i in range(n_imgs)]

    def _compute_batch_luts(
        self,
        d_gray,
        batch_n: int,
    ) -> np.ndarray:
        """Calcula N LUTs de ecualización en un solo viaje GPU→CPU."""
        h_hists = np.zeros((batch_n, HISTOGRAM_BINS), dtype=np.int32)
        d_hists = cuda.to_device(h_hists)
        shape = (d_gray.shape[1], d_gray.shape[2])
        blocks, threads = self._get_batch_grid_dims(shape, batch_n)
        # pylint: disable-next=possibly-used-before-assignment
        _histogram_batch_kernel[blocks, threads](d_gray, d_hists)
        cuda.synchronize()
        hists = d_hists.copy_to_host()
        return np.stack([compute_equalize_lut(hists[i]) for i in range(batch_n)])

    def _batch_equalize(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Ecualiza histograma en lote; un solo viaje D2H al final."""
        n = len(images)
        h, w = images[0].shape[0], images[0].shape[1]
        stacked_rgb = np.ascontiguousarray(np.stack(images))
        d_in = cuda.to_device(stacked_rgb)
        d_gray = cuda.device_array((n, h, w), dtype=np.uint8)
        blocks, threads = self._get_batch_grid_dims(images[0].shape, n)
        # pylint: disable-next=possibly-used-before-assignment
        _grayscale_batch_kernel[blocks, threads](d_in, d_gray)
        luts = self._compute_batch_luts(d_gray, n)
        d_luts = cuda.to_device(luts)
        d_out = cuda.device_array((n, h, w), dtype=np.uint8)
        # pylint: disable-next=possibly-used-before-assignment
        _map_batch_kernel[blocks, threads](d_gray, d_luts, d_out)
        cuda.synchronize()
        host = d_out.copy_to_host()
        return [host[i] for i in range(n)]

    def _handle_oom(self, image: np.ndarray, exc: Exception) -> None:
        """Loggea el OOM de GPU y notifica el fallback configurado."""
        _LOGGER.warning(
            'GPU OOM (%s), fallback a CPU: %r', image.shape, exc)
        if self._on_fallback is not None:
            self._on_fallback()

    def _process_on_gpu(
        self, image: np.ndarray, operation: str) -> np.ndarray:
        """Enruta la operación al conjunto de kernels correspondiente."""
        if operation == 'blur':
            return self._run_blur(image)
        if operation == 'equalize':
            return self._run_equalize(image)
        return self._run_simple(image, operation)

    def _get_grid_dims(self, shape: tuple) -> tuple[tuple, tuple]:
        """Calcula las dimensiones del grid y los bloques para CUDA."""
        threads = (CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK)
        blocks = (
            math.ceil(shape[0] / threads[0]),
            math.ceil(shape[1] / threads[1]),
        )
        return blocks, threads

    def _get_batch_grid_dims(
        self, shape: tuple, batch_n: int
    ) -> tuple[tuple, tuple]:
        """Calcula grid y bloques 3D para kernels batcheados (H, W, N)."""
        threads = (CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, 1)
        blocks = (
            math.ceil(shape[0] / CUDA_THREADS_PER_BLOCK),
            math.ceil(shape[1] / CUDA_THREADS_PER_BLOCK),
            batch_n,
        )
        return blocks, threads

    def _run_simple(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Ejecuta los kernels de un solo paso (grayscale/edges)."""
        if operation == "edges":
            image = _to_grayscale(image)
        d_image = cuda.to_device(image)
        d_output = cuda.to_device(np.zeros(image.shape[:2], dtype=np.uint8))
        blocks, threads = self._get_grid_dims(image.shape)
        _OPERATIONS_CUDA[operation][blocks, threads](d_image, d_output)
        cuda.synchronize()
        return d_output.copy_to_host()

    def _run_blur(self, image: np.ndarray) -> np.ndarray:
        """Aplica el desenfoque gaussiano conservando los 3 canales."""
        d_image = cuda.to_device(image)
        d_output = cuda.to_device(np.zeros(image.shape, dtype=np.uint8))
        d_mask = cuda.to_device(_gaussian_mask())
        blocks, threads = self._get_grid_dims(image.shape)
        # pylint: disable-next=possibly-used-before-assignment
        _blur_kernel[blocks, threads](d_image, d_output, d_mask, BLUR_RADIUS)
        cuda.synchronize()
        return d_output.copy_to_host()

    def _run_equalize(self, image: np.ndarray) -> np.ndarray:
        """Ecualiza el histograma sobre la imagen en escala de grises."""
        gray = _to_grayscale(image)
        histogram = self._compute_histogram(gray)
        lut = compute_equalize_lut(histogram)
        return self._apply_lut(gray, lut)

    def _compute_histogram(self, gray: np.ndarray) -> np.ndarray:
        """Calcula el histograma del gris en GPU con sumas atómicas."""
        d_gray = cuda.to_device(gray)
        d_hist = cuda.to_device(np.zeros(HISTOGRAM_BINS, dtype=np.int32))
        blocks, threads = self._get_grid_dims(gray.shape)
        # pylint: disable-next=possibly-used-before-assignment
        _histogram_kernel[blocks, threads](d_gray, d_hist)
        cuda.synchronize()
        return d_hist.copy_to_host()

    def _apply_lut(self, gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Mapea cada píxel del gris a través de la LUT en GPU."""
        d_gray = cuda.to_device(gray)
        d_lut = cuda.to_device(lut)
        d_output = cuda.to_device(np.zeros(gray.shape, dtype=np.uint8))
        blocks, threads = self._get_grid_dims(gray.shape)
        # pylint: disable-next=possibly-used-before-assignment
        _map_kernel[blocks, threads](d_gray, d_lut, d_output)
        cuda.synchronize()
        return d_output.copy_to_host()
