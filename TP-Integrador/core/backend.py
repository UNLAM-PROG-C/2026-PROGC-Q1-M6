"""Contratos de interfaz para los backends de procesamiento GPU/CPU."""

import numpy as np


VALID_OPERATIONS = ('grayscale', 'edges', 'blur', 'equalize')


class GPUBackend:
    """Interfaz comun para todos los backends de procesamiento.

    Define el contrato que deben cumplir CUDABackend, OpenCLBackend
    y CPUBackend (patron Strategy). El resto de la aplicacion solo
    interactua con esta interfaz.
    """

    def process(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """Aplica la operacion indicada a la imagen.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformacion a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado con el mismo shape que la entrada.

        Raises:
            NotImplementedError: Siempre; subclases deben implementar.
        """
        raise NotImplementedError


class CUDABackend(GPUBackend):
    """Backend para GPUs NVIDIA via Numba CUDA.

    Implementado en feature/gpu-cuda-opencl.
    """

    def process(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """Aplica la operacion usando kernels CUDA.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformacion a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado con el mismo shape que la entrada.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
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
        """Aplica la operacion usando kernels OpenCL.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformacion a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado con el mismo shape que la entrada.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError


class CPUBackend(GPUBackend):
    """Backend fallback sin GPU, usa ThreadPoolExecutor.

    Implementado en feature/cpu-pipeline-backend.
    """

    def process(
        self,
        image: np.ndarray,
        operation: str,
    ) -> np.ndarray:
        """Aplica la operacion usando Pillow / OpenCV en CPU.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformacion a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado con el mismo shape que la entrada.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError


def get_backend() -> GPUBackend:
    """Detecta el hardware disponible y retorna el backend optimo.

    Orden de prioridad: CUDA -> OpenCL -> CPU fallback.
    Implementado en feature/cpu-pipeline-backend.

    Returns:
        Instancia del backend disponible mas eficiente.

    Raises:
        NotImplementedError: Hasta que se complete la implementacion.
    """
    raise NotImplementedError
