"""Módulo de backends de procesamiento."""

from core.backend.base import (
    GPUBackend,
    VALID_OPERATIONS,
    CPU_BACKEND_NAME,
    CUDA_BACKEND_NAME,
    OPENCL_BACKEND_NAME,
)
from core.backend.cpu import CPUBackend
from core.backend.cuda import CUDABackend
from core.backend.opencl import OpenCLBackend
from core.backend.factory import get_backend

__all__ = [
    "GPUBackend",
    "CPUBackend",
    "CUDABackend",
    "OpenCLBackend",
    "get_backend",
    "VALID_OPERATIONS",
    "CPU_BACKEND_NAME",
    "CUDA_BACKEND_NAME",
    "OPENCL_BACKEND_NAME",
]
