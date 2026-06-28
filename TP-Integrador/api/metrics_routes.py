"""Endpoints de métricas: gráfico, resumen y exportación CSV."""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import api.metrics_state as metrics_state
from api.schemas import ChartPoint, OperationResult
from core.metrics import CPU_BACKEND, GPU_BACKEND
from pipeline.report_exporter import export_csv

BATCH_SIZE: int = 10
CSV_FILENAME: str = 'parallelvision_report.csv'
NO_RUN_DETAIL: str = 'No hay datos de métricas disponibles.'

metrics_router = APIRouter(prefix='/api/metrics')


def _require_collector():
    """Devuelve el colector activo o lanza 404."""
    collector = metrics_state.get_current()
    if collector is None:
        raise HTTPException(status_code=404, detail=NO_RUN_DETAIL)
    return collector


def _group_by_batch(times: list[float]) -> list[float]:
    """Promedia grupos de BATCH_SIZE tiempos en lotes secuenciales.

    Args:
        times: Lista de tiempos en ms ordenados cronológicamente.

    Returns:
        Lista de promedios, uno por lote de BATCH_SIZE tiempos.
    """
    result = []
    for i in range(0, len(times), BATCH_SIZE):
        batch = times[i:i + BATCH_SIZE]
        result.append(sum(batch) / len(batch))
    return result


def _op_speedup(cpu_ms: list[float], gpu_ms: list[float]) -> float:
    """Calcula speedup cpu_avg / gpu_avg; 0.0 si faltan datos GPU.

    Args:
        cpu_ms: Tiempos CPU de la operación.
        gpu_ms: Tiempos GPU de la operación.

    Returns:
        Speedup; 0.0 si gpu_ms está vacío.
    """
    if not gpu_ms or not cpu_ms:
        return 0.0
    return (sum(cpu_ms) / len(cpu_ms)) / (sum(gpu_ms) / len(gpu_ms))


@metrics_router.get('/chart')
def get_chart() -> list[ChartPoint]:
    """Serie temporal por lote de tiempos CPU vs GPU."""
    records = _require_collector().get_records()
    cpu_times = [r.elapsed_ms for r in records if r.backend == CPU_BACKEND]
    gpu_times = [r.elapsed_ms for r in records if r.backend == GPU_BACKEND]
    cpu_avgs = _group_by_batch(cpu_times)
    gpu_avgs = _group_by_batch(gpu_times)
    n = max(len(cpu_avgs), len(gpu_avgs), 0)
    return [
        ChartPoint(
            batch=i + 1,
            cpu_ms=cpu_avgs[i] if i < len(cpu_avgs) else 0.0,
            gpu_ms=gpu_avgs[i] if i < len(gpu_avgs) else 0.0,
        )
        for i in range(n)
    ]


@metrics_router.get('/summary')
def get_summary() -> list[OperationResult]:
    """Estadísticas por operación: promedios CPU/GPU y speedup."""
    records = _require_collector().get_records()
    ops: dict[str, tuple[list[float], list[float]]] = {}
    for r in records:
        cpu_lst, gpu_lst = ops.setdefault(r.operation, ([], []))
        if r.backend == CPU_BACKEND:
            cpu_lst.append(r.elapsed_ms)
        elif r.backend == GPU_BACKEND:
            gpu_lst.append(r.elapsed_ms)
    return [
        OperationResult(
            operation=op,
            cpu_avg_ms=sum(cpu) / len(cpu) if cpu else 0.0,
            gpu_avg_ms=sum(gpu) / len(gpu) if gpu else 0.0,
            speedup=_op_speedup(cpu, gpu),
        )
        for op, (cpu, gpu) in ops.items()
    ]


@metrics_router.get('/export')
def export_metrics() -> FileResponse:
    """Exporta los registros de la sesión como CSV descargable."""
    collector = _require_collector()
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, CSV_FILENAME)
    export_csv(collector, out_path)
    return FileResponse(
        out_path,
        media_type='text/csv',
        filename=CSV_FILENAME,
        headers={
            'Content-Disposition': f'attachment; filename="{CSV_FILENAME}"',
        },
    )
