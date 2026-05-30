"""Backend de procesamiento de imágenes en CPU.

Define el patrón Strategy con una única implementación concreta
(``CPUBackend``) y una fábrica ``get_backend()`` que la selecciona.
"""

from __future__ import annotations

import cv2
import numpy as np

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


def get_backend() -> CPUBackend:
    """Devuelve el backend de procesamiento activo.

    Returns:
        Instancia de CPUBackend.
    """
    # TODO: detección CUDA/OpenCL en futura iteración
    return CPUBackend()
