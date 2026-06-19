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
# Formula de conversion a grayscale:
CUDA_GRAYSCALE_RED_WEIGHT: float = 0.299
CUDA_GRAYSCALE_GREEN_WEIGHT: float = 0.587
CUDA_GRAYSCALE_BLUE_WEIGHT: float = 0.114

if _CUDA_AVAILABLE:

    @cuda.jit
    def _grayscale_kernel(image, output):
        """Kernel CUDA de conversión a escala de grises (un píxel por hilo)."""
        x, y = cuda.grid(2)
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
        x, y = cuda.grid(2)
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
            if magnitude > MAX_PIXEL_VALUE:
                magnitude = MAX_PIXEL_VALUE
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
