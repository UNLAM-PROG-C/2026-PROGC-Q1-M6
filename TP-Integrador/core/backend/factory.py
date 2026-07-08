"""Fábrica de backends (Strategy)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from core.backend.base import GPUBackend
from core.backend.cpu import CPUBackend
from core.backend.cuda import CUDABackend, _CUDA_AVAILABLE, cuda
from core.backend.opencl import OpenCLBackend, _OPENCL_AVAILABLE, cl


def _get_cuda_backend(
    on_fallback: Callable[[], None] | None = None,
) -> CUDABackend | None:
    """Intenta construir un CUDABackend; devuelve None si no hay CUDA."""
    if not _CUDA_AVAILABLE:
        return None
    try:
        if not cuda.is_available():
            return None
        backend = CUDABackend(on_fallback=on_fallback)
        logging.info("Backend seleccionado: CUDA")
        return backend
    except (cuda.cudadrv.error.CudaSupportError, RuntimeError) as exc:
        logging.info("CUDA no disponible: %s", exc)
        return None


def _get_opencl_backend(
    on_fallback: Callable[[], None] | None = None,
) -> OpenCLBackend | None:
    """Intenta construir un OpenCLBackend; devuelve None si no hay OpenCL."""
    if not _OPENCL_AVAILABLE:
        return None
    try:
        if not cl.get_platforms():
            return None
        backend = OpenCLBackend(on_fallback=on_fallback)
        logging.info("Backend seleccionado: OpenCL")
        return backend
    except (cl.Error, RuntimeError) as exc:
        logging.info("OpenCL no disponible: %s", exc)
        return None


def _get_cpu_backend() -> CPUBackend:
    """Construye y devuelve un CPUBackend."""
    backend = CPUBackend()
    logging.info("Backend seleccionado: CPU")
    return backend


def get_backend(
    on_fallback: Callable[[], None] | None = None,
) -> GPUBackend:
    """Devuelve el backend de procesamiento activo.

    Args:
        on_fallback: Callback opcional que los backends GPU invocan al
            caer a CPU por un OOM. CPUBackend lo ignora.

    Returns:
        El primer backend disponible en orden CUDA → OpenCL → CPU.
    """
    # backend = _get_cuda_backend(on_fallback)
    # if backend:
    #     return backend

    backend = _get_opencl_backend(on_fallback)
    if backend:
        return backend

    return _get_cpu_backend()
