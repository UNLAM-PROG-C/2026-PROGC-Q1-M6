"""Clases base y constantes para backends de procesamiento."""

from __future__ import annotations

import abc
import threading

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
MAX_PIXEL_VALUE: int = 255

CPU_BACKEND_NAME = "CPU"
CUDA_BACKEND_NAME = "CUDA"
OPENCL_BACKEND_NAME = "OpenCL"

MAX_GPU_CONCURRENT_BATCHES: int = 2
_gpu_semaphore = threading.Semaphore(MAX_GPU_CONCURRENT_BATCHES)


class GPUBackend(abc.ABC):
    """Interfaz común de los backends de procesamiento (Strategy)."""

    backend_name: str
    device_info: str

    @abc.abstractmethod
    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Aplica ``operation`` a ``image`` y devuelve el resultado."""
