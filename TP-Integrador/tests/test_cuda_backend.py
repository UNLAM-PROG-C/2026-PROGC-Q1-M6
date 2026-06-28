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
    with patch('core.backend.cuda.cuda.get_current_device'), \
         patch('core.backend.cuda.cuda.to_device') as mock_to_device, \
         patch('core.backend.cuda._OPERATIONS_CUDA', new_callable=dict) as mock_ops, \
         patch('core.backend.cuda.cuda.synchronize'):
         
        mock_d_array = MagicMock()
        mock_d_array.copy_to_host.return_value = mock_output
        mock_to_device.return_value = mock_d_array
        
        mock_kernel = MagicMock()
        mock_ops[operation] = mock_kernel
        
        backend = CUDABackend()
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

def test_cuda_memory_not_leaked(test_image):
    # This test is always mocked to specifically check if copy_to_host was called
    with patch('core.backend.cuda.cuda.get_current_device'), \
         patch('core.backend.cuda.cuda.to_device') as mock_to_device, \
         patch('core.backend.cuda._OPERATIONS_CUDA', new_callable=dict) as mock_ops, \
         patch('core.backend.cuda.cuda.synchronize'):
         
        mock_d_array = MagicMock()
        mock_to_device.return_value = mock_d_array
        
        mock_kernel = MagicMock()
        mock_ops['grayscale'] = mock_kernel
        
        backend = CUDABackend()
        backend.process(test_image, "grayscale")
        
        assert mock_d_array.copy_to_host.called

