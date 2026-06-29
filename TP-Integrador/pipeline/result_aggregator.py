"""Consumidor dedicado de resultados hacia MetricsCollector."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from core.metrics import MetricsCollector
from core.queue_manager import ImageQueue

_LOGGER = logging.getLogger(__name__)

RECORDS_PER_IMAGE_DEFAULT: int = 1


class ResultAggregator(threading.Thread):
    """Hilo dedicado que consolida resultados en el MetricsCollector."""

    def __init__(
        self,
        result_queue: ImageQueue,
        metrics: MetricsCollector,
        total_images: int = 0,
        on_record: Callable[[int, int], None] | None = None,
        records_per_image: int = RECORDS_PER_IMAGE_DEFAULT,
    ) -> None:
        """Inicializa el agregador con su cola y colector.

        Args:
            result_queue: Cola de resultados producida por los workers.
            metrics: Colector thread-safe de métricas.
            total_images: Cantidad total de imágenes a procesar.
            on_record: Callback opcional invocado tras cada imagen procesada.
            records_per_image: Records emitidos por imagen (2 en benchmark).
        """
        super().__init__(name='result-aggregator')
        self._result_queue = result_queue
        self._metrics = metrics
        self._total_images = total_images
        self._on_record = on_record
        self._records_per_image = records_per_image
        self._total_records = 0

    def run(self) -> None:
        """Consume resultados hasta recibir el sentinela None."""
        while True:
            result = self._result_queue.get()
            try:
                if result is None:
                    break
                self._metrics.record(*result)
                self._total_records += 1
                processed = self._total_records // self._records_per_image
                if (self._on_record
                        and self._total_records % self._records_per_image == 0):
                    self._on_record(processed, self._total_images)
            finally:
                self._result_queue.task_done()
        _LOGGER.info('Agregador de resultados finalizado')
