"""Backends de procesamiento de imágenes (patrón Strategy).

Define la interfaz común ``GPUBackend`` y sus implementaciones concretas
(``CUDABackend``, ``OpenCLBackend``, ``CPUBackend``), junto con la fábrica
``get_backend()`` que selecciona el backend disponible en orden
CUDA → OpenCL → CPU. Los imports de las librerías GPU están protegidos para
que el fallback a CPU funcione aunque numba o pyopencl no estén instalados.
"""

from __future__ import annotations

import abc
import math
import logging

import cv2
import numpy as np

try:
    from numba import cuda
    _CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    _CUDA_AVAILABLE = False

try:
    import pyopencl as cl
    _OPENCL_AVAILABLE = True
except ImportError:
    cl = None
    _OPENCL_AVAILABLE = False

VALID_OPERATIONS: tuple[str, ...] = (
    'grayscale',
    'edges',
    'blur',
    'equalize',
)
GAUSSIAN_KERNEL_SIZE: tuple[int, int] = (5, 5)
GAUSSIAN_SIGMA: float = 0.0
CANNY_LOW_THRESHOLD: int = 100
CANNY_HIGH_THRESHOLD: int = 200
MAX_PIXEL_VALUE: int = 255

CPU_BACKEND_NAME = "CPU"
CUDA_BACKEND_NAME = "CUDA"
OPENCL_BACKEND_NAME = "OpenCL"

# CONSTANTES CUDA
CUDA_THREADS_PER_BLOCK: int = 16
# Formula de conversion a grayscale:
CUDA_GRAYSCALE_RED_WEIGHT: float = 0.299
CUDA_GRAYSCALE_GREEN_WEIGHT: float = 0.587
CUDA_GRAYSCALE_BLUE_WEIGHT: float = 0.114


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte la imagen a escala de grises."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _detect_edges(image: np.ndarray) -> np.ndarray:
    """Detecta bordes con el algoritmo de Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)


def _apply_blur(image: np.ndarray) -> np.ndarray:
    """Aplica un desenfoque gaussiano a la imagen."""
    return cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)


def _equalize(image: np.ndarray) -> np.ndarray:
    """Ecualiza el histograma de la imagen en escala de grises."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


_OPERATIONS = {
    'grayscale': _to_grayscale,
    'edges': _detect_edges,
    'blur': _apply_blur,
    'equalize': _equalize,
}


# Los kernels llevan el decorador @cuda.jit en tiempo de carga, por eso se
# definen solo si numba/CUDA está disponible; si no, el dict queda vacío.
if _CUDA_AVAILABLE:

    @cuda.jit
    def _grayscale_kernel(image, output):
        """Kernel CUDA de conversión a escala de grises (un píxel por hilo)."""
        # Coordenadas (x, y) del pixel que procesa este hilo.
        x, y = cuda.grid(2)
        # Procesa solo si el pixel cae dentro de los limites de la imagen.
        if x < image.shape[0] and y < image.shape[1]:
            # Canales de color del pixel actual.
            r = image[x, y, 0]
            g = image[x, y, 1]
            b = image[x, y, 2]
            # Aplica la formula ponderada y guarda el resultado en la salida.
            output[x, y] = (
                CUDA_GRAYSCALE_RED_WEIGHT * r
                + CUDA_GRAYSCALE_GREEN_WEIGHT * g
                + CUDA_GRAYSCALE_BLUE_WEIGHT * b
            )

    @cuda.jit
    def _edges_kernel(image, output):
        """Kernel CUDA de detección de bordes (gradiente Sobel en device).

        Sobel calcula el cambio de intensidad entre píxeles vecinos.
        """
        x, y = cuda.grid(2)
        # Evita los bordes para no acceder a indices fuera de rango.
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


class GPUBackend(abc.ABC):
    """Interfaz común de los backends de procesamiento (Strategy)."""

    backend_name: str
    device_info: str

    @abc.abstractmethod
    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Aplica ``operation`` a ``image`` y devuelve el resultado."""


class CUDABackend(GPUBackend):
    """Procesa imágenes en GPU NVIDIA mediante kernels CUDA (Numba).

    Operaciones soportadas:
    - grayscale: 0.299*R + 0.587*G + 0.114*B por píxel en paralelo.
    - edges: gradiente Sobel sobre la imagen en escala de grises en device.
    """

    def __init__(self):
        device = cuda.get_current_device()
        self.backend_name = CUDA_BACKEND_NAME
        self.device_info = str(device)

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Ejecuta el kernel CUDA correspondiente sobre la imagen.

        Args:
            image: Array NumPy con la imagen de entrada.
            operation: Transformación a aplicar ('grayscale' o 'edges').

        Returns:
            Array NumPy con la imagen procesada.

        Raises:
            ValueError: Si operation no es una operación CUDA válida.
        """
        if operation not in _OPERATIONS_CUDA:
            raise ValueError(f'Unknown operation: {operation!r}')
        # edges recibe RGB → se convierte a gris → Sobel CUDA.
        if operation == "edges":
            image = _to_grayscale(image)
        # alloc en device + copia host→device.
        d_image = cuda.to_device(image)
        # alloc + copia del buffer de salida (inicializado en ceros).
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
        # copia device→host (alloc en host + copia).
        return d_output.copy_to_host()
        # Nota: Numba libera la memoria del device por GC; no hace falta free()
        #       explícito (a diferencia de PyCUDA, que requiere *_gpu.free()).


class OpenCLBackend(GPUBackend):
    """Esqueleto de backend OpenCL para GPU AMD/Intel (sin implementar)."""

    def __init__(self):
        device = cl.get_platforms()[0].get_devices()[0]
        self.backend_name = OPENCL_BACKEND_NAME
        self.device_info = device.name

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Procesa la imagen en GPU vía OpenCL.

        Raises:
            NotImplementedError: La implementación OpenCL es un esqueleto.
        """
        raise NotImplementedError("OpenCL backend no implementado todavía")


class CPUBackend(GPUBackend):
    """Aplica operaciones de transformación de imágenes en CPU."""

    def __init__(self):
        self.backend_name = CPU_BACKEND_NAME
        self.device_info = "CPU"

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Aplica la operación indicada a la imagen.

        Args:
            image: Array NumPy con la imagen de entrada.
            operation: Transformación a aplicar; debe pertenecer a
                VALID_OPERATIONS.

        Returns:
            Array NumPy con la imagen procesada.

        Raises:
            ValueError: Si operation no es una operación válida.
        """
        if operation not in VALID_OPERATIONS:
            raise ValueError(f'Unknown operation: {operation!r}')
        return _OPERATIONS[operation](image)


def _get_cuda_backend() -> CUDABackend | None:
    """Devuelve un CUDABackend si hay GPU NVIDIA disponible, si no None."""
    if not _CUDA_AVAILABLE:
        return None
    try:
        if not cuda.is_available():
            return None
        backend = CUDABackend()
        logging.info("Backend seleccionado: CUDA")
        return backend
    except (cuda.cudadrv.error.CudaSupportError, RuntimeError) as exc:
        logging.info("CUDA no disponible: %s", exc)
        return None


def _get_opencl_backend() -> OpenCLBackend | None:
    """Devuelve un OpenCLBackend si hay plataforma OpenCL, si no None."""
    if not _OPENCL_AVAILABLE:
        return None
    try:
        if not cl.get_platforms():
            return None
        backend = OpenCLBackend()
        logging.info("Backend seleccionado: OpenCL")
        return backend
    except (cl.Error, RuntimeError) as exc:
        logging.info("OpenCL no disponible: %s", exc)
        return None


def _get_cpu_backend() -> CPUBackend:
    """Devuelve siempre un CPUBackend (fallback final)."""
    backend = CPUBackend()
    logging.info("Backend seleccionado: CPU")
    return backend


def get_backend() -> GPUBackend:
    """Devuelve el backend de procesamiento activo.

    Selecciona en orden CUDA → OpenCL → CPU según el hardware disponible.

    Returns:
        Instancia de GPUBackend (CUDABackend, OpenCLBackend o CPUBackend).
    """
    backend = _get_cuda_backend()
    if backend:
        return backend

    backend = _get_opencl_backend()
    if backend:
        return backend

    return _get_cpu_backend()
