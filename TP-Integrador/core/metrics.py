"""Recolector thread-safe de métricas de rendimiento."""

from __future__ import annotations

import threading
from typing import NamedTuple

ZERO_THROUGHPUT: float = 0.0
NO_SPEEDUP: float = 0.0
CPU_BACKEND: str = 'cpu'
GPU_BACKEND: str = 'gpu'


class Record(NamedTuple):
    """Registro crudo de una imagen procesada."""

    image_name: str
    backend: str
    operation: str
    elapsed_ms: float


def _mean(values: list[float]) -> float:
    """Calcula el promedio de una lista no vacía de valores."""
    return sum(values) / len(values)


def _summarize(times: list[float]) -> dict:
    """Resume una lista de tiempos en estadísticas básicas."""
    return {
        'count': len(times),
        'avg_ms': _mean(times),
        'min_ms': min(times),
        'max_ms': max(times),
    }


class MetricsCollector:
    """Acumula tiempos de procesamiento por operación, thread-safe."""

    def __init__(self) -> None:
        """Inicializa el almacenamiento protegido por Lock."""
        self._lock: threading.Lock = threading.Lock()
        self._samples: dict[str, list[float]] = {}
        self._records: list[Record] = []
        self._total_count: int = 0

    def record(
        self,
        image_name: str,
        backend: str,
        operation: str,
        elapsed_ms: float,
    ) -> None:
        """Registra el tiempo de una imagen procesada.

        Args:
            image_name: Nombre del archivo de imagen procesado.
            backend: Backend que procesó la imagen (CPU_BACKEND, etc.).
            operation: Operación aplicada a la imagen.
            elapsed_ms: Tiempo de procesamiento en milisegundos.
        """
        with self._lock:
            self._records.append(
                Record(image_name, backend, operation, elapsed_ms))
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

    def get_records(self) -> list[Record]:
        """Devuelve una copia de los registros crudos de la sesión.

        Returns:
            Lista de tuplas (image_name, backend, operation, elapsed_ms).
        """
        with self._lock:
            return list(self._records)

    def get_speedup(self) -> float:
        """Calcula el speedup promedio de GPU respecto de CPU.

        Returns:
            Cociente entre el tiempo medio CPU y GPU; NO_SPEEDUP si
            falta alguno de los dos backends.
        """
        with self._lock:
            cpu = [r.elapsed_ms for r in self._records
                   if r.backend == CPU_BACKEND]
            gpu = [r.elapsed_ms for r in self._records
                   if r.backend == GPU_BACKEND]
            if not cpu or not gpu:
                return NO_SPEEDUP
            return _mean(cpu) / _mean(gpu)

    def reset(self) -> None:
        """Borra todas las métricas y registros acumulados."""
        with self._lock:
            self._samples.clear()
            self._records.clear()
            self._total_count = 0
