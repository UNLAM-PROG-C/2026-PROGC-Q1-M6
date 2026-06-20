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
        # pylint: disable-next=comparison-with-callable

        if x < image.shape[0] and y < image.shape[1]:
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
        x, y = cuda.grid(CUDA_GRID_DIM)  # pylint: disable=no-value-for-parameter
        # pylint: disable-next=comparison-with-callable
        if (EDGE_MARGIN <= x < image.shape[0] - EDGE_MARGIN and
                EDGE_MARGIN <= y < image.shape[1] - EDGE_MARGIN):
            gx = (
                -image[x-EDGE_MARGIN, y-EDGE_MARGIN] + image[x-EDGE_MARGIN, y+EDGE_MARGIN]
                - SOBEL_WEIGHT*image[x, y-EDGE_MARGIN] + SOBEL_WEIGHT*image[x, y+EDGE_MARGIN]
                - image[x+EDGE_MARGIN, y-EDGE_MARGIN] + image[x+EDGE_MARGIN, y+EDGE_MARGIN]
            )
            gy = (
                -image[x-EDGE_MARGIN, y-EDGE_MARGIN] - SOBEL_WEIGHT*image[x-EDGE_MARGIN, y] - image[x-EDGE_MARGIN, y+EDGE_MARGIN]
                + image[x+EDGE_MARGIN, y-EDGE_MARGIN] + SOBEL_WEIGHT*image[x+EDGE_MARGIN, y] + image[x+EDGE_MARGIN, y+EDGE_MARGIN]
            )
            magnitude = math.sqrt(gx*gx + gy*gy)
            output[x, y] = min(magnitude, MAX_PIXEL_VALUE)

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

    def _get_grid_dims(self, shape: tuple) -> tuple[tuple, tuple]:
        """Calcula las dimensiones del grid y los bloques para CUDA."""
        threads = (CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK)
        blocks = (
            math.ceil(shape[0] / threads[0]),
            math.ceil(shape[1] / threads[1]),
        )
        return blocks, threads

    def _run_kernel(self, image: np.ndarray, operation: str) -> np.ndarray:
        if operation == "edges":
            image = _to_grayscale(image)
        d_image = cuda.to_device(image)
        d_output = cuda.to_device(
            np.zeros(image.shape[:2], dtype=np.uint8)
        )
        
        blocks, threads = self._get_grid_dims(image.shape)
        _OPERATIONS_CUDA[operation][blocks, threads](d_image, d_output)
        
        cuda.synchronize()
        return d_output.copy_to_host()
