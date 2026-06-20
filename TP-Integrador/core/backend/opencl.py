"""Implementación del backend de procesamiento para OpenCL."""

from __future__ import annotations

import numpy as np

from core.backend.base import GPUBackend, OPENCL_BACKEND_NAME, _gpu_semaphore

try:
    import pyopencl as cl
    _OPENCL_AVAILABLE = True
except ImportError:
    cl = None
    _OPENCL_AVAILABLE = False


class OpenCLBackend(GPUBackend):
    """Esqueleto de backend OpenCL para GPU AMD/Intel (sin implementar)."""

    def __init__(self):
        device = cl.get_platforms()[0].get_devices()[0]
        self.backend_name = OPENCL_BACKEND_NAME
        self.device_info = device.name

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        with _gpu_semaphore:
            raise NotImplementedError("OpenCL backend no implementado todavía")
