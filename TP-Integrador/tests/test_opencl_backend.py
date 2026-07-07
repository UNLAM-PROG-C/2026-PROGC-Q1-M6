import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from core.backend.opencl import OpenCLBackend, _OPENCL_AVAILABLE

try:
    import pyopencl as cl
    platforms = cl.get_platforms() if _OPENCL_AVAILABLE else []
    HAS_OPENCL_HARDWARE = len(platforms) > 0 and len(platforms[0].get_devices()) > 0
except Exception:
    HAS_OPENCL_HARDWARE = False

@pytest.fixture
def test_image():
    return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

def _mocked_opencl_process(operation, mock_output):
    """Configura los mocks para simular la inicialización y proceso en OpenCL."""
    with patch('core.backend.opencl.cl.get_platforms') as mock_platforms, \
         patch('core.backend.opencl.cl.Context'), \
         patch('core.backend.opencl.cl.CommandQueue'), \
         patch('core.backend.opencl.cl.Program'), \
         patch('core.backend.opencl.cl.Kernel'), \
         patch('core.backend.opencl.cl.Buffer'), \
         patch('core.backend.opencl.cl.enqueue_copy') as mock_copy:
         
        mock_device = MagicMock()
        mock_device.name = "Mocked OpenCL Device"
        mock_platform = MagicMock()
        mock_platform.get_devices.return_value = [mock_device]
        mock_platforms.return_value = [mock_platform]
        
        # Simular que al copiar, modifica la memoria de output (el array numpy)
        def side_effect_copy(queue, dest, src):
            dest[:] = mock_output[:]
            mock_wait = MagicMock()
            return mock_wait
        mock_copy.side_effect = side_effect_copy
        
        backend = OpenCLBackend()
        # Mockeamos kernel para que no intente ejecutarlo
        backend.program = MagicMock()
        
        return backend.process(np.zeros((10, 10, 3), dtype=np.uint8), operation)

def test_opencl_grayscale_output_shape(test_image):
    if HAS_OPENCL_HARDWARE:
        backend = OpenCLBackend()
        result = backend.process(test_image, "grayscale")
    else:
        result = _mocked_opencl_process("grayscale", np.zeros((10, 10), dtype=np.uint8))
        
    assert result.shape == (10, 10)

def test_opencl_grayscale_output_dtype(test_image):
    if HAS_OPENCL_HARDWARE:
        backend = OpenCLBackend()
        result = backend.process(test_image, "grayscale")
    else:
        result = _mocked_opencl_process("grayscale", np.zeros((10, 10), dtype=np.uint8))

    assert result.dtype == np.uint8

def test_opencl_edges_output_shape(test_image):
    if HAS_OPENCL_HARDWARE:
        backend = OpenCLBackend()
        result = backend.process(test_image, "edges")
    else:
        result = _mocked_opencl_process("edges", np.zeros((10, 10), dtype=np.uint8))

    assert result.shape == (10, 10)

def test_opencl_process_invalid_operation():
    if HAS_OPENCL_HARDWARE:
        backend = OpenCLBackend()
        with pytest.raises(ValueError):
            backend.process(np.zeros((10, 10, 3), dtype=np.uint8), "invalid_op")
    else:
        with patch('core.backend.opencl.cl.get_platforms') as mock_platforms, \
             patch('core.backend.opencl.cl.Context'), \
             patch('core.backend.opencl.cl.CommandQueue'), \
             patch('core.backend.opencl.cl.Program'), \
             patch('core.backend.opencl.cl.Kernel'):
            mock_device = MagicMock()
            mock_platform = MagicMock()
            mock_platform.get_devices.return_value = [mock_device]
            mock_platforms.return_value = [mock_platform]

            backend = OpenCLBackend()
            with pytest.raises(ValueError):
                backend.process(
                    np.zeros((10, 10, 3), dtype=np.uint8), "invalid_op")

def test_opencl_context_initialized_once():
    # We always use mock for this test so we can count calls to cl.Context
    with patch('core.backend.opencl.cl.get_platforms') as mock_platforms, \
         patch('core.backend.opencl.cl.Context') as mock_context, \
         patch('core.backend.opencl.cl.CommandQueue'), \
         patch('core.backend.opencl.cl.Program'), \
         patch('core.backend.opencl.cl.Kernel'), \
         patch('core.backend.opencl.cl.Buffer'), \
         patch('core.backend.opencl.cl.enqueue_copy'):
         
        mock_device = MagicMock()
        mock_platform = MagicMock()
        mock_platform.get_devices.return_value = [mock_device]
        mock_platforms.return_value = [mock_platform]
        
        # Instantiate backend
        backend = OpenCLBackend()
        
        # Verify Context was created exactly once during init
        assert mock_context.call_count == 1
        
        # Call process
        backend.program = MagicMock() # mock the kernel
        backend.process(np.zeros((10, 10, 3), dtype=np.uint8), "grayscale")

        # Verify Context is STILL 1 (not created again during process)
        assert mock_context.call_count == 1


def test_opencl_process_batch_signature():
    """OpenCLBackend.process_batch tiene la firma correcta."""
    import inspect
    sig = inspect.signature(OpenCLBackend.process_batch)
    assert 'images' in sig.parameters
    assert 'operation' in sig.parameters


@pytest.mark.parametrize(
    'operation', ['grayscale', 'edges', 'blur', 'equalize'])
def test_opencl_process_batch_on_hardware(test_image, operation):
    """En hardware real, process_batch iguala al resultado de process."""
    if not HAS_OPENCL_HARDWARE:
        pytest.skip('No OpenCL hardware disponible')
    backend = OpenCLBackend()
    expected = backend.process(test_image, operation)
    results, per_ms = backend.process_batch(
        [test_image, test_image], operation)
    np.testing.assert_array_equal(expected, results[0])
    np.testing.assert_array_equal(expected, results[1])
    assert per_ms is not None and per_ms >= 0


def test_opencl_warmup_smoke():
    """warmup(op) no lanza excepciones en hardware real."""
    if not HAS_OPENCL_HARDWARE:
        pytest.skip('No OpenCL hardware disponible')
    backend = OpenCLBackend()
    for operation in ('grayscale', 'edges', 'blur', 'equalize'):
        backend.warmup(operation)
