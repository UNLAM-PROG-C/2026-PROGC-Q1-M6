"""Implementación del backend de procesamiento para GPU NVIDIA (CUDA)."""

from __future__ import annotations

import math
import os
import sys
import numpy as np

from core.backend.base import (
    GPUBackend,
    CUDA_BACKEND_NAME,
    MAX_PIXEL_VALUE,
    _gpu_semaphore,
)
from core.backend.cpu import _to_grayscale

try:
    from numba import cuda
    _CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    _CUDA_AVAILABLE = False


# CONSTANTES CUDA
CUDA_THREADS_PER_BLOCK: int = 16
# Formula de conversion a grayscale:
CUDA_GRAYSCALE_RED_WEIGHT: float = 0.299
CUDA_GRAYSCALE_GREEN_WEIGHT: float = 0.587
CUDA_GRAYSCALE_BLUE_WEIGHT: float = 0.114

if _CUDA_AVAILABLE:

    @cuda.jit
    def _grayscale_kernel(image, output):
        """Kernel CUDA de conversión a escala de grises (un píxel por hilo)."""
        x, y = cuda.grid(2)  # pylint: disable=no-value-for-parameter
        # pylint: disable-next=comparison-with-callable
        if x < image.shape[0] and y < image.shape[1]:
            r = image[x, y, 0]
            g = image[x, y, 1]
            b = image[x, y, 2]
            output[x, y] = (
                CUDA_GRAYSCALE_RED_WEIGHT * r
                + CUDA_GRAYSCALE_GREEN_WEIGHT * g
                + CUDA_GRAYSCALE_BLUE_WEIGHT * b
            )

    @cuda.jit
    def _edges_kernel(image, output):
        """Kernel CUDA de detección de bordes (gradiente Sobel en device)."""
        x, y = cuda.grid(2)  # pylint: disable=no-value-for-parameter
        # pylint: disable-next=comparison-with-callable
        if 1 <= x < image.shape[0] - 1 and 1 <= y < image.shape[1] - 1:
            gx = (
                -image[x-1, y-1] + image[x-1, y+1]
                - 2*image[x, y-1] + 2*image[x, y+1]
                - image[x+1, y-1] + image[x+1, y+1]
            )
            gy = (
                -image[x-1, y-1] - 2*image[x-1, y] - image[x-1, y+1]
                + image[x+1, y-1] + 2*image[x+1, y] + image[x+1, y+1]
            )
            magnitude = math.sqrt(gx*gx + gy*gy)
            magnitude = min(magnitude, MAX_PIXEL_VALUE)
            output[x, y] = magnitude

    _OPERATIONS_CUDA = {
        'grayscale': _grayscale_kernel,
        'edges': _edges_kernel,
    }
else:
    _OPERATIONS_CUDA = {}


class CUDABackend(GPUBackend):
    """Procesa imágenes en GPU NVIDIA mediante kernels CUDA (Numba)."""

    def __init__(self):
        device = cuda.get_current_device()
        self.backend_name = CUDA_BACKEND_NAME
        self.device_info = str(device)

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Aplica la operación a la imagen usando un kernel CUDA.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformación a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado con shape (H, W), dtype uint8.

        Raises:
            ValueError: Si operation no está en _OPERATIONS_CUDA.
        """
        if operation not in _OPERATIONS_CUDA:
            raise ValueError(f'Unknown operation: {operation!r}')
        with _gpu_semaphore:
            return self._run_kernel(image, operation)

    def _run_kernel(self, image: np.ndarray, operation: str) -> np.ndarray:
        if operation == "edges":
            image = _to_grayscale(image)
        d_image = cuda.to_device(image)
        d_output = cuda.to_device(
            np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        )
        threads = (CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK)
        blocks = (
            math.ceil(image.shape[0] / threads[0]),
            math.ceil(image.shape[1] / threads[1]),
        )
        _OPERATIONS_CUDA[operation][blocks, threads](d_image, d_output)
        cuda.synchronize()
        return d_output.copy_to_host()
