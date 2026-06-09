import numpy as np
import pytest

from numba import cuda

import core.backend as backend_module
from core.backend import CUDABackend
from unittest.mock import MagicMock
from unittest.mock import patch

cuda_available = pytest.mark.skipif(
    not cuda.is_available(),
    reason="CUDA no disponible"
)

@pytest.fixture
def test_image():
    return np.random.randint(
        0,
        256,
        (10, 10, 3),
        dtype=np.uint8
    )

@cuda_available
def test_cuda_grayscale_output_shape(test_image):

    backend = CUDABackend()

    result = backend.process(
        test_image,
        "grayscale"
    )

    assert result.shape == (10, 10)

@cuda_available
def test_cuda_grayscale_output_dtype(test_image):

    backend = CUDABackend()

    result = backend.process(
        test_image,
        "grayscale"
    )

    assert result.dtype == np.uint8

@cuda_available
def test_cuda_edges_output_shape(test_image):

    backend = CUDABackend()

    result = backend.process(
        test_image,
        "edges"
    )

    assert result.shape == (10, 10)

def test_cuda_process_invalid_operation():

    image = np.zeros(
        (10, 10, 3),
        dtype=np.uint8
    )

    backend = CUDABackend()

    with pytest.raises(ValueError):
        backend.process(
            image,
            "blur"
        )

@patch("core.backend.cuda.to_device")
@patch("core.backend.cuda.synchronize")
def test_cuda_memory_not_leaked(mock_sync,mock_to_device):

    image = np.zeros((10, 10, 3), dtype=np.uint8)

    fake_d_image = MagicMock()
    fake_d_output = MagicMock()

    mock_to_device.side_effect = [
        fake_d_image,
        fake_d_output
    ]

    fake_kernel = MagicMock()
    fake_kernel.__getitem__.return_value = MagicMock()

    original_kernel = backend_module._OPERATIONS_CUDA["grayscale"]

    try:
        backend_module._OPERATIONS_CUDA["grayscale"] = fake_kernel

        backend = CUDABackend()
        backend.process(image, "grayscale")

        fake_d_output.copy_to_host.assert_called_once()

    finally:
        backend_module._OPERATIONS_CUDA["grayscale"] = original_kernel