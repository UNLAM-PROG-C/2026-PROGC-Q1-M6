"""Worker de procesamiento concurrente de imágenes."""

from __future__ import annotations

import logging
import os
import threading
import time

import cv2

from core.backend.base import GPUBackend
from core.metrics import CPU_FALLBACK_BACKEND
from core.queue_manager import ImageQueue

try:
    from core.backend.cuda import _COMPUTE_TLS as _GPU_COMPUTE_TLS
except ImportError:
    _GPU_COMPUTE_TLS = None

MAX_WORKER_THREADS: int = 4
MS_PER_SECOND: float = 1000.0

_LOGGER = logging.getLogger(__name__)

# Estado por-hilo para etiquetar la imagen que cayó a CPU por OOM de GPU.
_FALLBACK_TLS = threading.local()


def _mark_fallback() -> None:
    """Marca que el hilo actual cayó a CPU durante el último process()."""
    _FALLBACK_TLS.triggered = True


class ProcessingWorker:
    """Consume rutas de imágenes y produce métricas de tiempo."""

    def __init__(
        self,
        input_queue: ImageQueue,
        result_queue: ImageQueue,
        backend: GPUBackend,
        operation: str,
        *,
        backend_label: str,
        io_queue: ImageQueue | None = None,
        bench_backend: GPUBackend | None = None,
        bench_label: str | None = None,
    ) -> None:
        """Inicializa el worker con sus colas, backend y operación.

        Args:
            input_queue: Cola de rutas de imágenes a procesar.
            result_queue: Cola donde se publican las métricas.
            backend: Backend de procesamiento activo.
            operation: Operación a aplicar a cada imagen.
            backend_label: Etiqueta del backend para las métricas.
            io_queue: Cola de guardado de imágenes (opcional).
            bench_backend: Backend secundario para benchmark (opcional).
            bench_label: Etiqueta del backend de benchmark (opcional).
        """
        self._input_queue = input_queue
        self._result_queue = result_queue
        self._backend = backend
        self._operation = operation
        self._backend_label = backend_label
        self._io_queue = io_queue
        self._bench_backend = bench_backend
        self._bench_label = bench_label
        if hasattr(backend, '_on_fallback'):
            backend._on_fallback = _mark_fallback

    def run(self) -> None:
        """Procesa imágenes hasta recibir el sentinela None."""
        while True:
            path = self._input_queue.get()
            try:
                if path is None:
                    break
                self._process_one(path)
            except Exception as e:
                _LOGGER.error("Worker error processing %s: %s", path, e)
            finally:
                self._input_queue.task_done()

    def _time_backend(
        self,
        image: object,
        backend: GPUBackend,
        label: str,
    ) -> tuple[str, float, object]:
        """Mide el tiempo de backend sobre image.

        Args:
            image: Array de imagen a procesar.
            backend: Backend a cronometrar.
            label: Etiqueta nominal del backend.

        Returns:
            Tupla (etiqueta_efectiva, ms, imagen_procesada).
        """
        _FALLBACK_TLS.triggered = False
        if _GPU_COMPUTE_TLS is not None:
            _GPU_COMPUTE_TLS.last_ms = None
        start = time.perf_counter()
        processed = backend.process(image, self._operation)
        elapsed_ms = (time.perf_counter() - start) * MS_PER_SECOND
        gpu_ms = getattr(_GPU_COMPUTE_TLS, 'last_ms', None)
        if gpu_ms is not None:
            elapsed_ms = gpu_ms
        actual = (CPU_FALLBACK_BACKEND
                  if getattr(_FALLBACK_TLS, 'triggered', False)
                  else label)
        return actual, elapsed_ms, processed

    def _enqueue_bench(self, image: object, name: str) -> None:
        """Encola el record del backend de benchmark si está configurado."""
        if self._bench_backend is None:
            return
        _, bench_ms, _ = self._time_backend(
            image, self._bench_backend, self._bench_label or '')
        self._result_queue.put(
            (name, self._bench_label, self._operation, bench_ms))

    def _process_one(self, path: str) -> None:
        """Procesa una imagen y encola su métrica de tiempo."""
        import numpy as np
        try:
            image = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None:
            _LOGGER.warning('Imagen ilegible: %s', path)
            return
        name = os.path.basename(path)
        label, elapsed_ms, processed = self._time_backend(
            image, self._backend, self._backend_label)
        if self._io_queue is not None:
            self._io_queue.put((name, self._operation, processed))
        self._result_queue.put((name, label, self._operation, elapsed_ms))
        self._enqueue_bench(image, name)
