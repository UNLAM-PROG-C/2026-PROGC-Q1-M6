from unittest.mock import Mock, patch
from core.backend import (get_backend,CUDABackend, OpenCLBackend, CPUBackend)
from core.backend.factory import (
    DISABLE_OPENCL_ENV_VAR,
    _get_opencl_backend,
)

@patch("core.backend.factory._get_opencl_backend")
@patch("core.backend.factory._get_cuda_backend")
def test_opencl_selected_when_cuda_fails(mock_cuda,mock_opencl):

    mock_cuda.return_value = None

    fake_backend = Mock()
    fake_backend.backend_name = "OpenCL"

    mock_opencl.return_value = fake_backend

    backend = get_backend()

    assert backend.backend_name == "OpenCL"

@patch("core.backend.factory._get_cuda_backend")
def test_returns_gpu_backend_instance(mock_cuda):

    fake_backend = Mock(spec=CUDABackend)

    mock_cuda.return_value = fake_backend

    backend = get_backend()

    assert backend is fake_backend

@patch("core.backend.factory._get_cuda_backend")
def test_cuda_selected_when_available(mock_cuda):

    fake_backend = Mock()
    fake_backend.backend_name = "CUDA"

    mock_cuda.return_value = fake_backend

    backend = get_backend()

    assert backend.backend_name == "CUDA"

@patch("core.backend.factory._get_opencl_backend")
@patch("core.backend.factory._get_cuda_backend")
def test_fallback_to_cpu_when_no_gpu(mock_cuda, mock_opencl):

    mock_cuda.return_value = None
    mock_opencl.return_value = None

    backend = get_backend()

    assert isinstance(backend,CPUBackend)

def test_backend_name_accessible():

    backend = CPUBackend()

    assert hasattr(backend,"backend_name" )

@patch.dict("os.environ", {DISABLE_OPENCL_ENV_VAR: "1"})
@patch("core.backend.factory._get_cuda_backend")
def test_opencl_skipped_when_env_var_set(mock_cuda):

    mock_cuda.return_value = None

    backend = get_backend()

    assert isinstance(backend, CPUBackend)

@patch.dict("os.environ", {DISABLE_OPENCL_ENV_VAR: "1"})
def test_get_opencl_backend_returns_none_when_disabled():

    assert _get_opencl_backend() is None

@patch("core.backend.opencl.cl")
def test_device_info_accessible(mock_cl):

    fake_device = Mock()
    fake_device.name = "Fake GPU"
    mock_cl.get_platforms.return_value = [Mock()]
    mock_cl.get_platforms.return_value[0].get_devices.return_value = [
        fake_device
    ]

    backend = OpenCLBackend()

    assert hasattr(backend, "device_info")
