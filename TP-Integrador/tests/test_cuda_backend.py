import numpy as np
import pytest
from numba import cuda

from core.backend import CUDABackend

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

@cuda_available
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
