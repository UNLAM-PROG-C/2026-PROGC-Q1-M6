"""Acumulador thread-safe de imágenes por shape para batching GPU."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from core.backend.base import MAX_GPU_BATCH_SIZE, NO_BATCH_TIME
from core.queue_manager import ImageQueue

if TYPE_CHECKING:
    from core.backend.base import GPUBackend

_LOGGER = logging.getLogger(__name__)


class GpuBatcher:
    """Agrupa imágenes por shape y las procesa en lotes en el backend GPU.

    Es el camino primario cuando hay GPU disponible: procesa, guarda (si
    se pasa io_queue) y registra un record por imagen en result_q.
    Thread-safe: múltiples workers pueden llamar a add() concurrentemente.
    """

    def __init__(
        self,
        result_q: ImageQueue,
        backend: GPUBackend,
        label: str,
        operation: str,
        io_queue: ImageQueue | None = None,
    ) -> None:
        """Inicializa el batcher con sus dependencias.

        Args:
            result_q: Cola donde se publican las métricas.
            backend: Backend GPU que implementa process_batch.
            label: Etiqueta del backend para las métricas.
            operation: Operación a aplicar en cada lote.
            io_queue: Cola de guardado de imágenes; None no guarda.
        """
        self._result_q = result_q
        self._backend = backend
        self._label = label
        self._operation = operation
        self._io_queue = io_queue
        self._buckets: dict[tuple, list[tuple[str, np.ndarray]]] = {}
        self._lock = threading.Lock()

    def warmup(self, operation: str) -> None:
        """Llama al warmup del backend si está disponible."""
        warmup_fn = getattr(self._backend, 'warmup', None)
        if warmup_fn is not None:
            warmup_fn(operation)

    def add(self, name: str, image: np.ndarray) -> None:
        """Agrega una imagen al bucket de su shape; flushea si está lleno.

        Args:
            name: Nombre de la imagen (basename del path).
            image: Array NumPy de la imagen cargada.
        """
        with self._lock:
            key = image.shape
            self._buckets.setdefault(key, []).append((name, image))
            if len(self._buckets[key]) >= MAX_GPU_BATCH_SIZE:
                bucket = self._buckets.pop(key)
            else:
                bucket = None
        if bucket is not None:
            self._flush(bucket)

    def flush_all(self) -> None:
        """Vacía todos los buckets parciales; llamar tras el shutdown.

        Garantiza que cada imagen produce exactamente un record.
        """
        with self._lock:
            buckets = list(self._buckets.values())
            self._buckets.clear()
        for bucket in buckets:
            self._flush(bucket)

    def _flush(self, bucket: list[tuple[str, np.ndarray]]) -> None:
        """Procesa el lote, guarda cada imagen y encola su record.

        Args:
            bucket: Lista de (name, image) del mismo shape.
        """
        names = [item[0] for item in bucket]
        images = [item[1] for item in bucket]
        results, per_img_ms = self._backend.process_batch(
            images, self._operation)
        if per_img_ms is NO_BATCH_TIME:
            _LOGGER.warning(
                'process_batch devolvió NO_BATCH_TIME; usando 0 ms')
            per_img_ms = 0.0
        for name, processed in zip(names, results):
            if self._io_queue is not None:
                self._io_queue.put((name, self._operation, processed))
            self._result_q.put(
                (name, self._label, self._operation, per_img_ms))
