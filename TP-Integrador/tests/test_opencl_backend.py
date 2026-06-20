import numpy as np
import pytest

from core.backend.opencl import OpenCLBackend, _OPENCL_AVAILABLE

opencl_available = pytest.mark.skipif(
    not _OPENCL_AVAILABLE,
    reason="OpenCL no disponible"
)

@pytest.fixture
def test_image():
    return np.random.randint(
        0,
        256,
        (10, 10, 3),
        dtype=np.uint8
    )

@opencl_available
def test_opencl_grayscale_output_shape(test_image):

    backend = OpenCLBackend()

    result = backend.process(
        test_image,
        "grayscale"
    )

    assert result.shape == (10, 10)

@opencl_available
def test_opencl_grayscale_output_dtype(test_image):

    backend = OpenCLBackend()

    result = backend.process(
        test_image,
        "grayscale"
    )

    assert result.dtype == np.uint8

@opencl_available
def test_opencl_edges_output_shape(test_image):

    backend = OpenCLBackend()

    result = backend.process(
        test_image,
        "edges"
    )

    assert result.shape == (10, 10)

@opencl_available
def test_opencl_process_invalid_operation():

    image = np.zeros(
        (10, 10, 3),
        dtype=np.uint8
    )

    backend = OpenCLBackend()

    with pytest.raises(NotImplementedError):
        backend.process(
            image,
            "blur"
        )
