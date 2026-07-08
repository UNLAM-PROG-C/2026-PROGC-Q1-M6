import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from numba import cuda
from core.backend.cuda import CUDABackend

# Variable para ver si hay hardware real
try:
    HAS_CUDA_HARDWARE = cuda.is_available() and len(cuda.gpus) > 0
except Exception:
    HAS_CUDA_HARDWARE = False

try:
    from numba.cuda.cudadrv.error import NvvmSupportError
except ImportError:
    class NvvmSupportError(Exception):
        pass

@pytest.fixture
def test_image():
    return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

def _mocked_cuda_process(operation, mock_output):
    """Configura los mocks para simular la inicialización y proceso en CUDA."""
    with patch('core.backend.cuda.cuda.get_current_device'):
        backend = CUDABackend()
        method_name = f'_run_{operation}'
        with patch.object(backend, method_name, return_value=mock_output):
            return backend.process(np.zeros((10, 10, 3), dtype=np.uint8), operation)

def test_cuda_grayscale_output_shape(test_image):
    if HAS_CUDA_HARDWARE:
        backend = CUDABackend()
        try:
            result = backend.process(test_image, "grayscale")
        except NvvmSupportError:
            pytest.skip("No GPU compute capabilities found")
    else:
        result = _mocked_cuda_process("grayscale", np.zeros((10, 10), dtype=np.uint8))

    assert result.shape == (10, 10)

def test_cuda_grayscale_output_dtype(test_image):
    if HAS_CUDA_HARDWARE:
        backend = CUDABackend()
        try:
            result = backend.process(test_image, "grayscale")
        except NvvmSupportError:
            pytest.skip("No GPU compute capabilities found")
    else:
        result = _mocked_cuda_process("grayscale", np.zeros((10, 10), dtype=np.uint8))

    assert result.dtype == np.uint8

def test_cuda_edges_output_shape(test_image):
    if HAS_CUDA_HARDWARE:
        backend = CUDABackend()
        try:
            result = backend.process(test_image, "edges")
        except NvvmSupportError:
            pytest.skip("No GPU compute capabilities found")
    else:
        result = _mocked_cuda_process("edges", np.zeros((10, 10), dtype=np.uint8))

    assert result.shape == (10, 10)

def test_cuda_process_invalid_operation():
    if HAS_CUDA_HARDWARE:
        backend = CUDABackend()
        with pytest.raises(ValueError):
            backend.process(np.zeros((10, 10, 3), dtype=np.uint8), "invalid_op")
    else:
        with patch('core.backend.cuda.cuda.get_current_device'):
            backend = CUDABackend()
            with pytest.raises(ValueError):
                backend.process(
                    np.zeros((10, 10, 3), dtype=np.uint8), "invalid_op")

def test_base_process_batch_default_returns_no_batch_time():
    """GPUBackend.process_batch default devuelve NO_BATCH_TIME."""
    from core.backend.base import NO_BATCH_TIME
    from core.backend.cpu import CPUBackend
    imgs = [np.zeros((5, 5, 3), dtype=np.uint8)] * 2
    results, per_ms = CPUBackend().process_batch(imgs, 'grayscale')
    assert len(results) == 2
    assert per_ms is NO_BATCH_TIME


def test_cuda_process_batch_signature():
    """CUDABackend.process_batch tiene la firma correcta."""
    import inspect
    sig = inspect.signature(CUDABackend.process_batch)
    assert 'images' in sig.parameters
    assert 'operation' in sig.parameters


def test_cuda_process_batch_on_hardware(test_image):
    """En hardware real, process_batch produce el mismo resultado que process."""
    if not HAS_CUDA_HARDWARE:
        pytest.skip('No CUDA hardware disponible')
    backend = CUDABackend()
    try:
        expected = backend.process(test_image, 'grayscale')
        results, per_ms = backend.process_batch(
            [test_image, test_image], 'grayscale')
    except NvvmSupportError:
        pytest.skip('No GPU compute capabilities found')
    np.testing.assert_array_equal(expected, results[0])
    np.testing.assert_array_equal(expected, results[1])
    assert per_ms is not None and per_ms >= 0


def test_cuda_memory_not_leaked(test_image):
    # This test is always mocked to specifically check if copy_to_host was called
    with patch('core.backend.cuda.cuda.get_current_device'), \
         patch('core.backend.cuda.cuda.to_device'), \
         patch('core.backend.cuda.cuda.device_array') as mock_device_array, \
         patch('core.backend.cuda.grayscale_kernel'), \
         patch('core.backend.cuda.cuda.synchronize'):
         
        mock_d_array = MagicMock()
        mock_device_array.return_value = mock_d_array
        
        backend = CUDABackend()
        backend.process(test_image, "grayscale")
        
        assert mock_d_array.copy_to_host.called

