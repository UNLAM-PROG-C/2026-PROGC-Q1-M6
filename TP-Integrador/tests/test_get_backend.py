from unittest.mock import Mock, patch
from core.backend import (get_backend,CUDABackend, OpenCLBackend, CPUBackend)

@patch("core.backend._get_opencl_backend")
@patch("core.backend._get_cuda_backend")
def test_opencl_selected_when_cuda_fails(mock_cuda,mock_opencl):

    mock_cuda.return_value = None

    fake_backend = Mock()
    fake_backend.backend_name = "OpenCL"

    mock_opencl.return_value = fake_backend

    backend = get_backend()

    assert backend.backend_name == "OpenCL"

@patch("core.backend._get_cuda_backend")
def test_returns_gpu_backend_instance(mock_cuda):

    fake_backend = Mock(spec=CUDABackend)

    mock_cuda.return_value = fake_backend

    backend = get_backend()

    assert backend is fake_backend

@patch("core.backend._get_cuda_backend")
def test_cuda_selected_when_available(mock_cuda):

    fake_backend = Mock()
    fake_backend.backend_name = "CUDA"

    mock_cuda.return_value = fake_backend

    backend = get_backend()

    assert backend.backend_name == "CUDA"

@patch("core.backend._get_opencl_backend")
@patch("core.backend._get_cuda_backend")
def test_fallback_to_cpu_when_no_gpu(mock_cuda, mock_opencl):

    mock_cuda.return_value = None
    mock_opencl.return_value = None

    backend = get_backend()

    assert isinstance(backend,CPUBackend)