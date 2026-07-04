"""Tests de GpuBatcher como camino primario (procesa, guarda, registra)."""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import GPU_BACKEND
from core.queue_manager import ImageQueue
from pipeline.gpu_batcher import GpuBatcher

OPERATION = 'grayscale'
FAKE_BATCH_MS = 2.0


class _FakeGPUBackend:
    """Backend GPU falso que solo invierte los canales de cada imagen."""

    backend_name = 'FakeGPU'
    device_info = 'fake'

    def process_batch(self, images, operation):
        """Devuelve las imágenes invertidas y un tiempo fijo por lote."""
        return [img + 1 for img in images], FAKE_BATCH_MS


@pytest.fixture(name='images')
def _images():
    shape = (4, 4, 3)
    return {
        'a.png': np.zeros(shape, dtype=np.uint8),
        'b.png': np.zeros(shape, dtype=np.uint8),
    }


def test_flush_all_saves_and_records_per_image(images):
    """Con io_queue, flush_all guarda cada imagen y emite su record gpu."""
    result_q = ImageQueue(10)
    io_q = ImageQueue(10)
    batcher = GpuBatcher(
        result_q, _FakeGPUBackend(), GPU_BACKEND, OPERATION, io_q)
    for name, image in images.items():
        batcher.add(name, image)
    batcher.flush_all()

    saved = []
    while not io_q.empty():
        saved.append(io_q.get())
    records = []
    while not result_q.empty():
        records.append(result_q.get())

    assert len(saved) == len(images)
    assert {item[0] for item in saved} == set(images)
    assert all(item[1] == OPERATION for item in saved)
    assert len(records) == len(images)
    assert all(r[1] == GPU_BACKEND for r in records)


def test_flush_all_without_io_queue_only_records(images):
    """Sin io_queue (--no-save), solo se registra tiempo, no se guarda."""
    result_q = ImageQueue(10)
    batcher = GpuBatcher(result_q, _FakeGPUBackend(), GPU_BACKEND, OPERATION)
    for name, image in images.items():
        batcher.add(name, image)
    batcher.flush_all()

    records = []
    while not result_q.empty():
        records.append(result_q.get())
    assert len(records) == len(images)
