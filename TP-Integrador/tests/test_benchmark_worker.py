"""Tests del modo benchmark del ProcessingWorker."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from core.backend.cpu import CPUBackend
from core.metrics import CPU_BACKEND
from core.queue_manager import ImageQueue
from pipeline.worker import ProcessingWorker

OPERATION = 'blur'
BENCH_LABEL = 'gpu'


class _FakeGPUBackend:
    """Backend GPU falso para tests de benchmark (sólo mide tiempo)."""

    backend_name = 'FakeGPU'
    device_info = 'fake'

    def process(self, image, operation):
        """Devuelve la imagen sin modificarla."""
        return image


@pytest.fixture(name='result_queue')
def _result_queue():
    return ImageQueue(20)


@pytest.fixture(name='fake_image_path')
def _fake_image_path(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = str(tmp_path / 'test.jpg')
    cv2.imwrite(path, img)
    return path


def test_benchmark_worker_emits_two_records(result_queue, fake_image_path):
    """En modo benchmark, _process_one emite un record cpu y uno gpu."""
    input_q = ImageQueue(10)
    worker = ProcessingWorker(
        input_q, result_queue, CPUBackend(), OPERATION,
        backend_label=CPU_BACKEND,
        bench_backend=_FakeGPUBackend(),
        bench_label=BENCH_LABEL,
    )
    worker._process_one(fake_image_path)
    records = []
    while not result_queue.empty():
        records.append(result_queue.get())
    assert len(records) == 2
    labels = {r[1] for r in records}
    assert CPU_BACKEND in labels
    assert BENCH_LABEL in labels


def test_single_backend_worker_emits_one_record(result_queue, fake_image_path):
    """Sin bench_backend, _process_one emite exactamente un record."""
    input_q = ImageQueue(10)
    worker = ProcessingWorker(
        input_q, result_queue, CPUBackend(), OPERATION,
        backend_label=CPU_BACKEND,
    )
    worker._process_one(fake_image_path)
    records = []
    while not result_queue.empty():
        records.append(result_queue.get())
    assert len(records) == 1
    assert records[0][1] == CPU_BACKEND
