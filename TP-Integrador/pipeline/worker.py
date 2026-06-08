"""Worker de procesamiento concurrente de imágenes."""

from __future__ import annotations

import logging
import os
import time

import cv2

from core.backend import CPUBackend
from core.queue_manager import ImageQueue

MAX_WORKER_THREADS: int = 4
MS_PER_SECOND: float = 1000.0

_LOGGER = logging.getLogger(__name__)


class ProcessingWorker:
    """Consume rutas de imágenes y produce métricas de tiempo."""

    def __init__(
        self,
        input_queue: ImageQueue,
        result_queue: ImageQueue,
        backend: CPUBackend,
        operation: str,
        *,
        backend_label: str,
    ) -> None:
        """Inicializa el worker con sus colas, backend y operación.

        Args:
            input_queue: Cola de rutas de imágenes a procesar.
            result_queue: Cola donde se publican las métricas.
            backend: Backend de procesamiento activo.
            operation: Operación a aplicar a cada imagen.
            backend_label: Etiqueta del backend para las métricas.
        """
        self._input_queue = input_queue
        self._result_queue = result_queue
        self._backend = backend
        self._operation = operation
        self._backend_label = backend_label

    def run(self) -> None:
        """Procesa imágenes hasta recibir el sentinela None."""
        while True:
            path = self._input_queue.get()
            try:
                if path is None:
                    break
                self._process_one(path)
            finally:
                self._input_queue.task_done()

    def _process_one(self, path: str) -> None:
        """Procesa una imagen y encola su métrica de tiempo."""
        image = cv2.imread(path)
        if image is None:
            _LOGGER.warning('Imagen ilegible: %s', path)
            return
        start = time.perf_counter()
        self._backend.process(image, self._operation)
        elapsed_ms = (time.perf_counter() - start) * MS_PER_SECOND
        name = os.path.basename(path)
        self._result_queue.put(
            (name, self._backend_label, self._operation, elapsed_ms))
