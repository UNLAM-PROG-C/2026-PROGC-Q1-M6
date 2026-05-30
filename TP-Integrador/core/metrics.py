"""Recolector thread-safe de métricas de rendimiento."""

from __future__ import annotations

import threading

ZERO_THROUGHPUT: float = 0.0


def _summarize(times: list[float]) -> dict:
    """Resume una lista de tiempos en estadísticas básicas."""
    return {
        'count': len(times),
        'avg_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
    }


class MetricsCollector:
    """Acumula tiempos de procesamiento por operación, thread-safe."""

    def __init__(self) -> None:
        """Inicializa el almacenamiento protegido por Lock."""
        self._lock: threading.Lock = threading.Lock()
        self._samples: dict[str, list[float]] = {}
        self._total_count: int = 0

    def record(
        self,
        image_name: str,
        operation: str,
        elapsed_ms: float,
    ) -> None:
        """Registra el tiempo de una imagen procesada.

        Args:
            image_name: Nombre del archivo de imagen procesado.
            operation: Operación aplicada a la imagen.
            elapsed_ms: Tiempo de procesamiento en milisegundos.
        """
        # image_name forma parte del contrato; el resumen agrega por
        # operación, por lo que el nombre no se almacena.
        del image_name
        with self._lock:
            self._samples.setdefault(operation, []).append(elapsed_ms)
            self._total_count += 1

    def get_summary(self) -> dict:
        """Devuelve estadísticas agregadas por operación.

        Returns:
            Dict por operación con count, avg_ms, min_ms y max_ms.
        """
        with self._lock:
            return {
                operation: _summarize(times)
                for operation, times in self._samples.items()
            }

    def get_throughput(self, total_seconds: float) -> float:
        """Calcula el throughput global en imágenes por segundo.

        Args:
            total_seconds: Tiempo total transcurrido en segundos.

        Returns:
            Imágenes procesadas por segundo; 0.0 si no hubo tiempo.
        """
        with self._lock:
            if total_seconds <= ZERO_THROUGHPUT:
                return ZERO_THROUGHPUT
            return self._total_count / total_seconds
