import numpy as np
from core.backend import CUDABackend

#verificar que la funcion trabaja sobre solo un pixel, y que el resultado es correcto.
def test_grayscale_single_pixel():

    image = np.array(
        [[[255, 0, 0]]],
        dtype=np.uint8
    )

    backend = CUDABackend()

    result = backend.grayscale(image)

    expected = int(
        0.299 * 255 +
        0.587 * 0 +
        0.114 * 0
    )

    assert result[0, 0] == expected

#verificar que la funcion detecta bordes en una imagen con un borde vertical.
def test_grayscale_shape():

    image = np.array(
        [
            [[255,0,0], [0,255,0]],
            [[0,0,255], [255,255,255]]
        ],
        dtype=np.uint8
    )

    backend = CUDABackend()

    result = backend.grayscale(image)

    assert result.shape == (2, 2)

def test_edges_detectar_borde_vertical():

    #se crea una imagen de 5x5 con un borde vertical en el medio (columna 2)
    image = np.array(
        [
            [0,0,0,255,255],
            [0,0,0,255,255],
            [0,0,0,255,255],
            [0,0,0,255,255],
            [0,0,0,255,255]
        ],
        dtype=np.uint8
    )

    backend = CUDABackend()

    result = backend.edges(image)

    assert result.max() > 0