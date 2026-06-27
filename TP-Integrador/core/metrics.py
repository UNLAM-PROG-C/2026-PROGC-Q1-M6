"""Recolector thread-safe de métricas de rendimiento."""

from __future__ import annotations

import threading
import time
from typing import NamedTuple

ZERO_THROUGHPUT: float = 0.0
NO_SPEEDUP: float = 0.0
CPU_BACKEND: str = 'cpu'
GPU_BACKEND: str = 'gpu'
STATS_WINDOW_SIZE: int = 10
NO_TOTAL: int = 0
ZERO_ELAPSED: float = 0.0
MIN_WINDOW_FOR_RATE: int = 2


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


def _imgs_per_sec(stamps: list[float]) -> float:
    """Calcula el rate en imágenes/segundo con ventana deslizante."""
    if len(stamps) < MIN_WINDOW_FOR_RATE:
        return ZERO_THROUGHPUT
    span = stamps[-1] - stamps[0]
    if span == ZERO_ELAPSED:
        return ZERO_THROUGHPUT
    return (len(stamps) - 1) / span


def _eta_seconds(total: int, processed: int, rate: float) -> float:
    """Estima segundos restantes; 0.0 si rate es cero (evita división)."""
    if rate <= ZERO_THROUGHPUT:
        return ZERO_ELAPSED
    return max(total - processed, 0) / rate


def _speedup_from_records(records: list[Record]) -> float:
    """Calcula el speedup GPU/CPU sin adquirir lock."""
    cpu = [r.elapsed_ms for r in records if r.backend == CPU_BACKEND]
    gpu = [r.elapsed_ms for r in records if r.backend == GPU_BACKEND]
    if not cpu or not gpu:
        return NO_SPEEDUP
    return _mean(cpu) / _mean(gpu)


def _compute_live_stats(
    stamps: list[float],
    records: list[Record],
    processed: int,
    total: int,
    start: float | None,
) -> dict[str, int | float | str]:
    """Arma el snapshot de métricas en vivo con datos ya copiados."""
    rate = _imgs_per_sec(stamps)
    elapsed = (time.monotonic() - start) if start is not None else ZERO_ELAPSED
    return {
        'processed_count': processed,
        'total_count': total,
        'imgs_per_sec': rate,
        'eta_seconds': _eta_seconds(total, processed, rate),
        'speedup_factor': _speedup_from_records(records),
        'elapsed_seconds': elapsed,
    }


class MetricsCollector:
    """Acumula tiempos de procesamiento por operación, thread-safe."""

    def __init__(self) -> None:
        """Inicializa el almacenamiento protegido por Lock."""
        self._lock: threading.Lock = threading.Lock()
        self._samples: dict[str, list[float]] = {}
        self._records: list[Record] = []
        self._total_count: int = 0
        self._timestamps: list[float] = []
        self._session_total: int = NO_TOTAL
        self._start_time: float | None = None

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
            now = time.monotonic()
            if self._start_time is None:
                self._start_time = now
            self._timestamps.append(now)
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
            records = list(self._records)
        return _speedup_from_records(records)

    def set_total_count(self, total: int) -> None:
        """Establece el total de imágenes de la sesión.

        Args:
            total: Cantidad total de imágenes a procesar.
        """
        with self._lock:
            self._session_total = total

    def get_live_stats(self) -> dict[str, int | float | str]:
        """Snapshot thread-safe del pipeline con 6 métricas en vivo."""
        with self._lock:
            stamps = list(self._timestamps[-STATS_WINDOW_SIZE:])
            records = list(self._records)
            processed = len(self._records)
            total = self._session_total
            start = self._start_time
        return _compute_live_stats(stamps, records, processed, total, start)

    def reset(self) -> None:
        """Borra todas las métricas y registros acumulados."""
        with self._lock:
            self._samples.clear()
            self._records.clear()
            self._total_count = 0
            self._timestamps.clear()
            self._session_total = NO_TOTAL
            self._start_time = None
