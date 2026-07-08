"""Tests de integración end-to-end del pipeline GPU (issues #24/#25/#27)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.backend import VALID_OPERATIONS, CPUBackend, get_backend
from core.backend.cuda import CUDABackend
from core.metrics import CPU_FALLBACK_BACKEND, MetricsCollector
from core.queue_manager import ImageQueue
from pipeline.image_loader import enqueue_paths, scan_folder
from pipeline.result_aggregator import ResultAggregator
from pipeline.worker import ProcessingWorker

IMAGE_COUNT: int = 5
IMAGE_SIZE: int = 200
MAX_PIPELINE_SECONDS: float = 60.0
GPU_LABEL: str = 'gpu'
FALLBACK_IMAGE_COUNT: int = 2
SINGLE_WORKER: int = 1
DUAL_WORKER: int = 2


def _write_images(folder, count: int, size: int) -> None:
    """Escribe ``count`` imágenes RGB aleatorias de ``size`` px en folder."""
    for index in range(count):
        image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f'img_{index}.png'), image)


def _drive_workers(folder, backend, operation, workers, queues) -> None:
    """Lanza los workers en un pool y encola las rutas de ``folder``."""
    in_q, out_q = queues
    executor = ThreadPoolExecutor(max_workers=workers)
    for _ in range(workers):
        worker = ProcessingWorker(
            in_q, out_q, backend, operation, backend_label=GPU_LABEL)
        executor.submit(worker.run)
    enqueue_paths(scan_folder(str(folder)), in_q, workers)
    executor.shutdown(wait=True)


def _run_pipeline(folder, backend, operation, workers) -> MetricsCollector:
    """Ejecuta el pipeline real sobre ``folder`` y devuelve sus métricas."""
    metrics = MetricsCollector()
    capacity = IMAGE_COUNT + workers
    queues = (ImageQueue(capacity), ImageQueue(capacity))
    aggregator = ResultAggregator(queues[1], metrics)
    aggregator.start()
    _drive_workers(folder, backend, operation, workers, queues)
    queues[1].put(None)
    aggregator.join()
    return metrics


def _assert_functional_equivalence(backend) -> None:
    """Compara shape y dtype GPU vs CPU para cada operación válida."""
    cpu = CPUBackend()
    image = np.random.randint(
        0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for operation in VALID_OPERATIONS:
        gpu_out = backend.process(image, operation)
        cpu_out = cpu.process(image, operation)
        assert gpu_out.shape == cpu_out.shape
        assert gpu_out.dtype == cpu_out.dtype


@pytest.mark.skipif(
    isinstance(get_backend(), CPUBackend),
    reason='Requiere GPU real (CUDA u OpenCL)')
def test_full_gpu_pipeline(tmp_path):
    """El pipeline GPU procesa 5 imágenes en menos de 60 segundos."""
    backend = get_backend()
    _write_images(tmp_path, IMAGE_COUNT, IMAGE_SIZE)
    start = time.perf_counter()
    metrics = _run_pipeline(tmp_path, backend, 'blur', DUAL_WORKER)
    elapsed = time.perf_counter() - start
    assert elapsed < MAX_PIPELINE_SECONDS
    assert len(metrics.get_records()) == IMAGE_COUNT
    _assert_functional_equivalence(backend)


def _build_sim_cuda_backend() -> CUDABackend:
    """Construye un CUDABackend sin requerir hardware GPU real."""
    with patch('core.backend.cuda.cuda.get_current_device'):
        return CUDABackend()


def test_gpu_fallback_on_oom(tmp_path):
    """Ante un OOM mockeado, el pipeline cae a CPU y lo registra."""
    backend = _build_sim_cuda_backend()
    _write_images(tmp_path, FALLBACK_IMAGE_COUNT, IMAGE_SIZE)
    with patch.object(backend, '_run_grayscale', side_effect=MemoryError):
        metrics = _run_pipeline(tmp_path, backend, 'grayscale', SINGLE_WORKER)
    records = metrics.get_records()
    assert len(records) == FALLBACK_IMAGE_COUNT
    assert all(r.backend == CPU_FALLBACK_BACKEND for r in records)


def test_gpu_fallback_matches_cpu_result():
    """El resultado del fallback es idéntico al de CPUBackend."""
    backend = _build_sim_cuda_backend()
    image = np.random.randint(
        0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    expected = CPUBackend().process(image, 'grayscale')
    with patch.object(backend, '_run_grayscale', side_effect=MemoryError):
        result = backend.process(image, 'grayscale')
    assert np.array_equal(result, expected)
