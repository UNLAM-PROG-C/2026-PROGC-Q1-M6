import numpy as np
import pytest

from core.backend import CPUBackend

#Imagen de prueba. Tamaño 10x10 con 3 canales RGB
@pytest.fixture
def test_image():
    return np.random.randint(
        0,
        256,
        (10, 10, 3),
        dtype=np.uint8
    )

def test_grayscale_output_shape(test_image):

    backend = CPUBackend()

    result = backend.process(
        test_image,
        "grayscale"
    )

    assert result.shape == (10, 10)

def test_grayscale_output_dtype(test_image):

    backend = CPUBackend()

    result = backend.process(
        test_image,
        "grayscale"
    )

    assert result.dtype == np.uint8

def test_edges_output_shape(test_image):

    backend = CPUBackend()

    result = backend.process(
        test_image,
        "edges"
    )

    assert result.shape == (10, 10)

def test_blur_output_shape(test_image):

    backend = CPUBackend()

    result = backend.process(
        test_image,
        "blur"
    )

    assert result.shape == test_image.shape

def test_equalize_output_shape(test_image):

    backend = CPUBackend()

    result = backend.process(
        test_image,
        "equalize"
    )

    assert result.shape == (10, 10)


def test_invalid_operation_raises(test_image):

    backend = CPUBackend()

    with pytest.raises(ValueError):backend.process(test_image, "invalid")


def test_all_operations_complete(test_image):

    backend = CPUBackend()

    operations = [
        "grayscale",
        "edges",
        "blur",
        "equalize"
    ]

    for operation in operations:

        result = backend.process(test_image,operation)

        assert result is not None