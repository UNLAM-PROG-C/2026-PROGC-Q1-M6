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

    Thread-safe: múltiples workers pueden llamar a add() concurrentemente.
    Cada flush produce un record GPU por imagen en la cola de resultados.
    """

    def __init__(
        self,
        result_q: ImageQueue,
        bench_backend: GPUBackend,
        bench_label: str,
        operation: str,
    ) -> None:
        """Inicializa el batcher con las dependencias de benchmark.

        Args:
            result_q: Cola donde se publican las métricas GPU.
            bench_backend: Backend GPU que implementa process_batch.
            bench_label: Etiqueta del backend GPU para las métricas.
            operation: Operación a aplicar en cada lote.
        """
        self._result_q = result_q
        self._backend = bench_backend
        self._bench_label = bench_label
        self._operation = operation
        self._buckets: dict[tuple, list[tuple[str, np.ndarray]]] = {}
        self._lock = threading.Lock()

    def warmup(self, operation: str) -> None:
        """Llama al warmup del backend bench si está disponible."""
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

        Garantiza que cada imagen produce exactamente un record GPU.
        """
        with self._lock:
            buckets = list(self._buckets.values())
            self._buckets.clear()
        for bucket in buckets:
            self._flush(bucket)

    def _flush(self, bucket: list[tuple[str, np.ndarray]]) -> None:
        """Procesa el lote y encola un record GPU por imagen.

        Args:
            bucket: Lista de (name, image) del mismo shape.
        """
        names = [item[0] for item in bucket]
        images = [item[1] for item in bucket]
        _, per_img_ms = self._backend.process_batch(images, self._operation)
        if per_img_ms is NO_BATCH_TIME:
            _LOGGER.warning('process_batch devolvió NO_BATCH_TIME; usando 0 ms')
            per_img_ms = 0.0
        for name in names:
            self._result_q.put(
                (name, self._bench_label, self._operation, per_img_ms))
