"""Tests de MetricsCollector: correctitud y acceso concurrente."""

from __future__ import annotations

import threading

from core.metrics import (
    CPU_BACKEND,
    GPU_BACKEND,
    NO_SPEEDUP,
    ZERO_THROUGHPUT,
    MetricsCollector,
)

OPERATION: str = 'grayscale'
OTHER_OPERATION: str = 'blur'
IMAGE_NAME: str = 'sample.png'
SINGLE_ELAPSED_MS: float = 12.5
ELAPSED_A_MS: float = 10.0
ELAPSED_B_MS: float = 20.0
ELAPSED_C_MS: float = 30.0
EXPECTED_AVG_AB_MS: float = 15.0
CPU_ELAPSED_MS: float = 100.0
GPU_ELAPSED_MS: float = 10.0
EXPECTED_SPEEDUP: float = 10.0
SAMPLE_SECONDS: float = 1.0
EXPECTED_SINGLE_COUNT: int = 1
THREAD_COUNT: int = 10
RECORDS_PER_THREAD: int = 100
TOTAL_RECORDS: int = 1000
READ_ATTEMPTS: int = 50


def test_record_single_entry() -> None:
    """Un registro aparece en el resumen con count e tiempo correctos."""
    collector = MetricsCollector()
    collector.record(IMAGE_NAME, CPU_BACKEND, OPERATION, SINGLE_ELAPSED_MS)
    summary = collector.get_summary()
    assert summary[OPERATION]['count'] == EXPECTED_SINGLE_COUNT
    assert summary[OPERATION]['avg_ms'] == SINGLE_ELAPSED_MS


def test_record_multiple_operations() -> None:
    """El resumen calcula el promedio por operación."""
    collector = MetricsCollector()
    collector.record(IMAGE_NAME, CPU_BACKEND, OPERATION, ELAPSED_A_MS)
    collector.record(IMAGE_NAME, CPU_BACKEND, OPERATION, ELAPSED_B_MS)
    collector.record(IMAGE_NAME, CPU_BACKEND, OTHER_OPERATION, ELAPSED_C_MS)
    summary = collector.get_summary()
    assert summary[OPERATION]['avg_ms'] == EXPECTED_AVG_AB_MS
    assert summary[OTHER_OPERATION]['avg_ms'] == ELAPSED_C_MS


def test_speedup_calculation() -> None:
    """El speedup es el cociente del tiempo medio CPU sobre GPU."""
    collector = MetricsCollector()
    collector.record(IMAGE_NAME, CPU_BACKEND, OPERATION, CPU_ELAPSED_MS)
    collector.record(IMAGE_NAME, GPU_BACKEND, OPERATION, GPU_ELAPSED_MS)
    assert collector.get_speedup() == EXPECTED_SPEEDUP


def test_speedup_no_data() -> None:
    """Sin ambos backends el speedup es NO_SPEEDUP, no una excepción."""
    collector = MetricsCollector()
    collector.record(IMAGE_NAME, CPU_BACKEND, OPERATION, CPU_ELAPSED_MS)
    assert collector.get_speedup() == NO_SPEEDUP


def test_reset_clears_data() -> None:
    """reset() deja el colector sin métricas ni registros."""
    collector = MetricsCollector()
    collector.record(IMAGE_NAME, CPU_BACKEND, OPERATION, SINGLE_ELAPSED_MS)
    collector.reset()
    assert collector.get_summary() == {}
    assert collector.get_throughput(SAMPLE_SECONDS) == ZERO_THROUGHPUT
    assert not collector.get_records()


def _record_batch(
    collector: MetricsCollector,
    barrier: threading.Barrier,
) -> None:
    """Espera la barrera y graba RECORDS_PER_THREAD registros."""
    barrier.wait()
    for index in range(RECORDS_PER_THREAD):
        collector.record(
            f'img-{index}.png', CPU_BACKEND, OPERATION, SINGLE_ELAPSED_MS)


def _build_threads(
    collector: MetricsCollector,
    barrier: threading.Barrier,
) -> list[threading.Thread]:
    """Crea THREAD_COUNT hilos que graban en el colector."""
    return [
        threading.Thread(target=_record_batch, args=(collector, barrier))
        for _ in range(THREAD_COUNT)
    ]


def test_concurrent_record_10_threads() -> None:
    """10 hilos × 100 registros producen 1000 sin pérdidas."""
    collector = MetricsCollector()
    barrier = threading.Barrier(THREAD_COUNT)
    threads = _build_threads(collector, barrier)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(collector.get_records()) == TOTAL_RECORDS
    assert collector.get_throughput(SAMPLE_SECONDS) == TOTAL_RECORDS


def test_get_summary_during_recording() -> None:
    """get_summary() durante la grabación no lanza ni corrompe datos."""
    collector = MetricsCollector()
    barrier = threading.Barrier(THREAD_COUNT)
    threads = _build_threads(collector, barrier)
    for thread in threads:
        thread.start()
    for _ in range(READ_ATTEMPTS):
        assert isinstance(collector.get_summary(), dict)
    for thread in threads:
        thread.join()
    assert len(collector.get_records()) == TOTAL_RECORDS
