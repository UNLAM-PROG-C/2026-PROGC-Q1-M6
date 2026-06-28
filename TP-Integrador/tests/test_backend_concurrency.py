import threading
import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from core.backend.base import MAX_GPU_CONCURRENT_BATCHES
from core.backend.cuda import CUDABackend

def test_semaphore_limits_concurrent_access():
    """
    Verifica que el semáforo _gpu_semaphore limite la cantidad de ejecuciones concurrentes
    en la GPU al máximo permitido por MAX_GPU_CONCURRENT_BATCHES.
    """
    active_threads = 0
    max_active_threads = 0
    lock = threading.Lock()
    
    def mock_process_on_gpu(*args, **kwargs):
        nonlocal active_threads, max_active_threads
        with lock:
            active_threads += 1
            if active_threads > max_active_threads:
                max_active_threads = active_threads
                
        # Simulamos trabajo en la GPU para dar tiempo a otros hilos a intentar entrar
        time.sleep(0.05)
        
        with lock:
            active_threads -= 1
            
        return np.zeros((10, 10), dtype=np.uint8)

    # Mockeamos toda la inicialización de hardware para que pueda correr sin GPU
    with patch('core.backend.cuda.cuda.get_current_device'), \
         patch.object(
             CUDABackend, '_process_on_gpu',
             side_effect=mock_process_on_gpu):
         
        backend = CUDABackend()
        
        threads = []
        # Lanzamos 10 hilos simultáneos
        for _ in range(10):
            t = threading.Thread(
                target=backend.process, 
                args=(np.zeros((10, 10, 3), dtype=np.uint8), "grayscale")
            )
            threads.append(t)
            
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        # Verificamos que el pico de hilos concurrentes en _process_on_gpu no superó el límite
        assert max_active_threads <= MAX_GPU_CONCURRENT_BATCHES
        # Verificamos que al menos sí entró en ejecución
        assert max_active_threads > 0
