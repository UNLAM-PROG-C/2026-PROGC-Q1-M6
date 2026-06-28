"""Implementación del backend de procesamiento para GPU NVIDIA (CUDA)."""

from __future__ import annotations

import logging
import math
import os
import sys
from collections.abc import Callable

import numpy as np

from core.backend.base import (
    GPUBackend,
    CUDA_BACKEND_NAME,
    MAX_PIXEL_VALUE,
    VALID_OPERATIONS,
    BLUR_RADIUS,
    GAUSSIAN_BLUR_WEIGHTS_1D,
    HISTOGRAM_BINS,
    compute_equalize_lut,
    _gpu_semaphore,
)
from core.backend.cpu import CPUBackend, _to_grayscale

_LOGGER = logging.getLogger(__name__)

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
CUDA_THREADS_PER_BLOCK: int = 16
CUDA_GRID_DIM: int = 2
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
        x, y = cuda.grid(CUDA_GRID_DIM)  # pylint: disable=no-value-for-parameter
        rows, cols = image.shape[0], image.shape[1]
        m, sw = EDGE_MARGIN, SOBEL_WEIGHT
        # pylint: disable-next=comparison-with-callable
        if m <= x < rows - m and m <= y < cols - m:
            gx = (
                -int(image[x-m, y-m]) + int(image[x-m, y+m])
                - sw*int(image[x, y-m]) + sw*int(image[x, y+m])
                - int(image[x+m, y-m]) + int(image[x+m, y+m])
            )
            gy = (
                -int(image[x-m, y-m]) - sw*int(image[x-m, y])
                - int(image[x-m, y+m]) + int(image[x+m, y-m])
                + sw*int(image[x+m, y]) + int(image[x+m, y+m])
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
                return self._process_on_gpu(image, operation)
            except _GPU_OOM_ERRORS as exc:
                self._handle_oom(image, exc)
                return self._cpu_fallback.process(image, operation)

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
