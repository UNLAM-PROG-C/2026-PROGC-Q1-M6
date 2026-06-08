"""Backend de procesamiento de imágenes en CPU.

Define el patrón Strategy con una única implementación concreta
(``CPUBackend``) y una fábrica ``get_backend()`` que la selecciona.
"""

from __future__ import annotations
import math
import logging
import cv2
import numpy as np
from numba import cuda

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

    backend = CPUBackend()
    logging.info("Backend seleccionado: CPU")
    return backend

_OPERATIONS = {
    'grayscale': _to_grayscale,
    'edges': _detect_edges,
    'blur': _apply_blur,
    'equalize': _equalize,
}

#a continuacion, el backend CUDA para realizar cada funcion.
@cuda.jit
def grayscale_kernel(image, output):
    '''Usar pycuda para implementar el kernel de conversión a escala de grises.'''
    x, y = cuda.grid(2) #Obtiene las coordenadas del pixel actual en la imagen
    if x < image.shape[0] and y < image.shape[1]: #Pregunto si el pixel se encuentra dentro de los limites de la imagen.
        #Obtiene el valor de canal de cada color del pixel (r,g,b)
        r = image[x, y, 0]  
        g = image[x, y, 1]
        b = image[x, y, 2]
        output[x, y] = CUDA_GRAYSCALE_RED_WEIGHT * r + CUDA_GRAYSCALE_GREEN_WEIGHT * g + CUDA_GRAYSCALE_BLUE_WEIGHT * b #Aplica la formula de conversion a grayscale y guarda el resultado en la imagen de salida.

@cuda.jit
def edges_kernel(image, output):
    # implementar gradiente Sobel sobre la imagen en device
    # Sobel calcula el cambio de intensidad entre píxeles vecinos.

    x, y = cuda.grid(2)

    #verifica que el pixel no se encuentre en los bordes de la imagen para evitar acceder a indices fuera de rango
    if 1 <= x < image.shape[0]-1 and 1 <= y < image.shape[1]-1: 

        gx = (
            -image[x-1, y-1] + image[x-1, y+1]
            -2*image[x,   y-1] + 2*image[x,   y+1]
            -image[x+1, y-1] + image[x+1, y+1]
        )

        gy = (
            -image[x-1, y-1] -2*image[x-1, y] -image[x-1, y+1]
            +image[x+1, y-1] +2*image[x+1, y] +image[x+1, y+1]
        )

        magnitude = math.sqrt(gx*gx + gy*gy)

        if magnitude > 255:
            magnitude = 255

        output[x, y] = magnitude


class CUDABackend:
    '''
    Las operaciones que debe realizar son
    - grayscale — kernel CUDA: 0.299*R + 0.587*G + 0.114*B por píxel en paralelo
    - edges — implementar gradiente Sobel sobre la imagen en device, o llamar a OpenCV luego de copy_to_host (documentar elección)
    '''

    # funcion para ejecutar un kernel CUDA (o grayscale o edges) sobre la imagen dada y devolver el resultado
    def _execute_kernel(self, image, kernel):

        d_image = cuda.to_device(image)

        # Buffer limpio para evitar memoria residual
        d_output = cuda.to_device(
            np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        )

        threads = (CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK) #Número de hilos por bloque en cada dimensión (x,y)

        blocks = (
            math.ceil(image.shape[0] / threads[0]),
            math.ceil(image.shape[1] / threads[1])
        )

        kernel[blocks, threads](d_image, d_output) # Ejecuta el kernel recibido como parámetro
        cuda.synchronize()

        return d_output.copy_to_host()

    def grayscale(self, image):
        return self._execute_kernel(image, grayscale_kernel)

    def edges(self, image):
        return self._execute_kernel(image, edges_kernel)



class CPUBackend:
    """Aplica operaciones de transformación de imágenes en CPU."""

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


def _get_cuda_backend():
    try:
        from numba import cuda

        if not cuda.is_available():
            return None

        backend = CUDABackend()
        logging.info("Backend seleccionado: CUDA")
        return backend

    except Exception as exc:
        logging.info("CUDA no disponible: %s", exc)
        return None

def _get_opencl_backend():
    try:
        import pyopencl as cl

        if not cl.get_platforms():
            return None

        backend = OpenCLBackend()
        logging.info("Backend seleccionado: OpenCL")
        return backend

    except Exception as exc:
        logging.info("OpenCL no disponible: %s", exc)
        return None

def _get_cpu_backend():
    backend = CPUBackend()
    logging.info("Backend seleccionado: CPU")
    return backend

def get_backend() -> CPUBackend:
    """Devuelve el backend de procesamiento activo.

    Returns:
        Instancia de CPUBackend.
    """
    backend = _get_cuda_backend()
    if backend:
        return backend

    backend = _get_opencl_backend()
    if backend:
        return backend

    return _get_cpu_backend()
