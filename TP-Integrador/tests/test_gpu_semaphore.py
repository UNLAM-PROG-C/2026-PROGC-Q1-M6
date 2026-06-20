"""Tests del semáforo GPU: concurrencia limitada a MAX_GPU_CONCURRENT_BATCHES."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numba import cuda

from core.backend import (
    CUDABackend,
    MAX_GPU_CONCURRENT_BATCHES,
    _gpu_semaphore,
)

cuda_available = pytest.mark.skipif(
    not cuda.is_available(),
    reason="CUDA no disponible"
)

NUM_THREADS = 10


def test_semaphore_limits_concurrency():
    """El semáforo del módulo no permite más de MAX_GPU_CONCURRENT_BATCHES entradas simultáneas."""
    counter_lock = threading.Lock()
    active = 0
    max_observed = 0

    def worker():
        nonlocal active, max_observed
        with _gpu_semaphore:
            with counter_lock:
                active += 1
                if active > max_observed:
                    max_observed = active
            # Simula trabajo dentro de la región crítica.
            with counter_lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_observed <= MAX_GPU_CONCURRENT_BATCHES


@cuda_available
def test_concurrent_cuda_process_no_crash():
    """10 hilos llamando a CUDABackend.process() simultáneamente no crashean."""
    image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(CUDABackend().process, image, "grayscale")
            for _ in range(NUM_THREADS)
        ]
        results = [f.result() for f in futures]

    for result in results:
        assert result.shape == (10, 10)
