"""Contratos de interfaz para los backends de procesamiento de imágenes.

Define el patrón Strategy: GPUBackend como interfaz común y tres
implementaciones concretas (CUDA, OpenCL, CPU). La función get_backend()
selecciona automáticamente el mejor backend disponible en runtime.
"""

import numpy as np

VALID_OPERATIONS: tuple[str, ...] = (
    'grayscale',
    'edges',
    'blur',
    'equalize',
)


class GPUBackend:
    """Interfaz común para todos los backends de procesamiento.

    Implementa el patrón Strategy: el resto del pipeline interactúa
    únicamente con esta interfaz, sin conocer el hardware subyacente.
    """

    def process(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """Aplica la operación indicada a la imagen.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformación a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado con el mismo shape que la entrada.

        Raises:
            NotImplementedError: Implementar en subclases concretas.
        """
        raise NotImplementedError


class CUDABackend(GPUBackend):
    """Backend para GPUs NVIDIA via Numba CUDA kernels.

    Implementado en feature/gpu-cuda-opencl.
    """

    def process(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """Aplica la operación con kernels CUDA. Ver GPUBackend.process.

        Raises:
            NotImplementedError: Implementado en feature/gpu-cuda-opencl.
        """
        raise NotImplementedError


class OpenCLBackend(GPUBackend):
    """Backend para GPUs AMD e Intel via PyOpenCL.

    Implementado en feature/gpu-cuda-opencl.
    """

    def process(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """Aplica la operación con kernels OpenCL. Ver GPUBackend.process.

        Raises:
            NotImplementedError: Implementado en feature/gpu-cuda-opencl.
        """
        raise NotImplementedError


class CPUBackend(GPUBackend):
    """Backend fallback para máquinas sin GPU (ThreadPoolExecutor).

    Implementado en feature/cpu-pipeline-backend.
    """

    def process(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """Aplica la operación usando CPU. Ver GPUBackend.process.

        Raises:
            NotImplementedError: Implementado en feature/cpu-pipeline-backend.
        """
        raise NotImplementedError


def get_backend() -> GPUBackend:
    """Detecta el hardware disponible y retorna el backend óptimo.

    Orden de detección: CUDA → OpenCL → CPU (fallback).

    Returns:
        Instancia del backend más apropiado para el hardware actual.

    Raises:
        NotImplementedError: Implementado en feature/cpu-pipeline-backend.
    """
    raise NotImplementedError
