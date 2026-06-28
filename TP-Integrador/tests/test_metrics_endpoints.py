"""Tests de los endpoints /api/metrics/* con FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.metrics_state as metrics_state
from api.main import app
from core.metrics import CPU_BACKEND, GPU_BACKEND, MetricsCollector

client = TestClient(app)

OPERATION = 'blur'
N_IMAGES = 10


def _seed_metrics() -> MetricsCollector:
    """Crea un MetricsCollector con records cpu y gpu para N_IMAGES."""
    mc = MetricsCollector()
    mc.set_total_count(N_IMAGES)
    for i in range(N_IMAGES):
        mc.record(f'img{i:02d}.jpg', CPU_BACKEND, OPERATION, float(10 + i))
        mc.record(f'img{i:02d}.jpg', GPU_BACKEND, OPERATION, float(2 + i * 0.5))
    return mc


@pytest.fixture(autouse=True)
def _seed_and_clear():
    """Siembra metrics_state antes de cada test y lo limpia al salir."""
    metrics_state.set_current(_seed_metrics(), '/tmp')
    yield
    metrics_state.clear()


def test_chart_returns_batches():
    """GET /api/metrics/chart devuelve lotes con campos batch/cpu_ms/gpu_ms."""
    resp = client.get('/api/metrics/chart')
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0
    assert {'batch', 'cpu_ms', 'gpu_ms'} <= data[0].keys()


def test_summary_has_speedup():
    """GET /api/metrics/summary devuelve speedup > 1 con CPU más lento."""
    resp = client.get('/api/metrics/summary')
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    row = data[0]
    assert row['operation'] == OPERATION
    assert row['cpu_avg_ms'] > row['gpu_avg_ms'] > 0
    assert row['speedup'] > 1.0


def test_export_returns_csv():
    """GET /api/metrics/export devuelve un CSV con cabecera y registros."""
    resp = client.get('/api/metrics/export')
    assert resp.status_code == 200
    assert 'text/csv' in resp.headers['content-type']
    lines = resp.text.strip().split('\n')
    assert lines[0].startswith('image_name')
    assert len(lines) > 1


def test_chart_404_without_run():
    """GET /api/metrics/chart devuelve 404 cuando no hay run activo."""
    metrics_state.clear()
    resp = client.get('/api/metrics/chart')
    assert resp.status_code == 404
