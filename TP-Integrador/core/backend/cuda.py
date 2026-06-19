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

# pylint: disable=broad-exception-caught, protected-access
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
                from numba.cuda.cudadrv import libs
                paths = cuda_paths.get_cuda_paths()
                if os.path.isdir(bin_x64) and paths['cudalib_dir'].info != bin_x64:
                    paths['cudalib_dir'] = cuda_paths._env_path_tuple(
                        paths['cudalib_dir'].by, bin_x64
                    )
                
                import glob
                # Patch for Numba missing CUDA 13+ support
                _orig_get_cudalib = libs.get_cudalib
                def _patched_get_cudalib(lib, static=False):
                    if lib == 'cudart' and sys.platform == 'win32':
                        libdir = paths['cudalib_dir'].info
                        if libdir and os.path.isdir(libdir):
                            dlls = glob.glob(os.path.join(libdir, 'cudart64_*.dll'))
                            if dlls:
                                return max(dlls)
                    return _orig_get_cudalib(lib, static)
                libs.get_cudalib = _patched_get_cudalib

                from numba.cuda.cudadrv.nvvm import NVVM
                _orig_supported_ccs = NVVM.supported_ccs
                @property
                def _patched_supported_ccs(self):
                    ccs = _orig_supported_ccs.fget(self)
                    if not ccs:
                        return ((5, 0), (5, 2), (5, 3), (6, 0), (6, 1), (6, 2), 
                                (7, 0), (7, 2), (7, 5), (8, 0), (8, 6), (8, 7), 
                                (8, 9), (9, 0))
                    return ccs
                NVVM.supported_ccs = _patched_supported_ccs

                if os.path.isdir(nvvm_bin_x64) and not paths['nvvm'].info:
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
        # pylint: disable=no-value-for-parameter
        x, y = cuda.grid(CUDA_GRID_DIM)
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
        x, y = cuda.grid(CUDA_GRID_DIM)
        if (EDGE_MARGIN <= x < image.shape[0] - EDGE_MARGIN and
                EDGE_MARGIN <= y < image.shape[1] - EDGE_MARGIN):
            gx = (
                -int(image[x-EDGE_MARGIN, y-EDGE_MARGIN]) + int(image[x-EDGE_MARGIN, y+EDGE_MARGIN])
                - SOBEL_WEIGHT*int(image[x, y-EDGE_MARGIN]) + SOBEL_WEIGHT*int(image[x, y+EDGE_MARGIN])
                - int(image[x+EDGE_MARGIN, y-EDGE_MARGIN]) + int(image[x+EDGE_MARGIN, y+EDGE_MARGIN])
            )
            gy = (
                -int(image[x-EDGE_MARGIN, y-EDGE_MARGIN]) - SOBEL_WEIGHT*int(image[x-EDGE_MARGIN, y]) - int(image[x-EDGE_MARGIN, y+EDGE_MARGIN])
                + int(image[x+EDGE_MARGIN, y-EDGE_MARGIN]) + SOBEL_WEIGHT*int(image[x+EDGE_MARGIN, y]) + int(image[x+EDGE_MARGIN, y+EDGE_MARGIN])
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
        self.backend_name = CUDA_BACKEND_NAME
        try:
            device = cuda.get_current_device()
            self.device_info = str(device)
        except AttributeError:
            self.device_info = "CUDA Simulator"

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
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
