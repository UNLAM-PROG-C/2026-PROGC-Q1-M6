"""Clases base y constantes para backends de procesamiento."""

from __future__ import annotations

import abc
import threading

import numpy as np

OP_GRAYSCALE: str = 'grayscale'
OP_EDGES: str = 'edges'
OP_BLUR: str = 'blur'
OP_EQUALIZE: str = 'equalize'

VALID_OPERATIONS: tuple[str, ...] = (
    OP_GRAYSCALE,
    OP_EDGES,
    OP_BLUR,
    OP_EQUALIZE,
)
GAUSSIAN_KERNEL_SIZE: tuple[int, int] = (5, 5)
GAUSSIAN_SIGMA: float = 0.0
CANNY_LOW_THRESHOLD: int = 100
CANNY_HIGH_THRESHOLD: int = 200
MAX_PIXEL_VALUE: int = 255
WARMUP_IMAGE_SIZE: int = 32
RGB_CHANNELS: int = 3
GPU_THREADS_PER_BLOCK: int = 16

# Constantes comunes de transformaciones matemáticas
GRAYSCALE_R: float = 0.299
GRAYSCALE_G: float = 0.587
GRAYSCALE_B: float = 0.114
EDGE_MARGIN: int = 1
SOBEL_WEIGHT: float = 2.0

# Constantes del desenfoque gaussiano (kernel 5x5 separable).
BLUR_KERNEL_SIZE: int = 5
BLUR_RADIUS: int = 2
GAUSSIAN_BLUR_WEIGHTS_1D: tuple[float, ...] = (
    0.0625,
    0.25,
    0.375,
    0.25,
    0.0625,
)

# Constantes de la ecualización de histograma.
HISTOGRAM_BINS: int = 256


def compute_equalize_lut(histogram: np.ndarray) -> np.ndarray:
    """Construye la LUT de ecualización a partir del histograma.

    Args:
        histogram: Conteo de píxeles por nivel de gris (HISTOGRAM_BINS).

    Returns:
        Tabla de mapeo uint8 (HISTOGRAM_BINS) que ecualiza el histograma.
    """
    cdf = histogram.cumsum()
    nonzero = cdf[cdf > 0]
    cdf_min = nonzero[0] if nonzero.size else 0
    span = cdf[-1] - cdf_min
    if span <= 0:
        return np.zeros(HISTOGRAM_BINS, dtype=np.uint8)
    lut = (cdf - cdf_min) * MAX_PIXEL_VALUE / span
    return np.clip(lut, 0, MAX_PIXEL_VALUE).astype(np.uint8)

CPU_BACKEND_NAME = "CPU"
CUDA_BACKEND_NAME = "CUDA"
OPENCL_BACKEND_NAME = "OpenCL"

MAX_GPU_CONCURRENT_BATCHES: int = 2
MAX_GPU_BATCH_SIZE: int = 32
NO_BATCH_TIME: float | None = None
_gpu_semaphore = threading.Semaphore(MAX_GPU_CONCURRENT_BATCHES)


class GPUBackend(abc.ABC):
    """Interfaz común de los backends de procesamiento (Strategy)."""

    backend_name: str
    device_info: str

    @abc.abstractmethod
    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Aplica ``operation`` a ``image`` y devuelve el resultado."""

    def process_batch(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> tuple[list[np.ndarray], float | None]:
        """Procesa un lote de imágenes del mismo shape.

        Args:
            images: Lista de arrays del mismo shape, dtype uint8.
            operation: Transformación a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Tupla (resultados, per_image_ms). Devuelve NO_BATCH_TIME si el
            backend no amortiza el lote; los backends GPU lo sobreescriben.
        """
        results = [self.process(img, operation) for img in images]
        return results, NO_BATCH_TIME
