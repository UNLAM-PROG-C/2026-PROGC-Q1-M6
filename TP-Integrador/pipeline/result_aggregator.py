"""Consumidor dedicado de resultados hacia MetricsCollector."""

from __future__ import annotations

import logging
import threading

from core.metrics import MetricsCollector
from core.queue_manager import ImageQueue

_LOGGER = logging.getLogger(__name__)


class ResultAggregator(threading.Thread):
    """Hilo dedicado que consolida resultados en el MetricsCollector."""

    def __init__(
        self,
        result_queue: ImageQueue,
        metrics: MetricsCollector,
    ) -> None:
        """Inicializa el agregador con su cola y colector.

        Args:
            result_queue: Cola de resultados producida por los workers.
            metrics: Colector thread-safe de métricas.
        """
        super().__init__(name='result-aggregator')
        self._result_queue = result_queue
        self._metrics = metrics

    def run(self) -> None:
        """Consume resultados hasta recibir el sentinela None."""
        while True:
            result = self._result_queue.get()
            try:
                if result is None:
                    break
                self._metrics.record(*result)
            finally:
                self._result_queue.task_done()
        _LOGGER.info('Agregador de resultados finalizado')
