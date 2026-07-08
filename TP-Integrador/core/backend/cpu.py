"""Implementación del backend de procesamiento para CPU."""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from core.backend.base import (
    GPUBackend,
    CPU_BACKEND_NAME,
    VALID_OPERATIONS,
    MAX_PIXEL_VALUE,
    BLUR_RADIUS,
    GAUSSIAN_BLUR_WEIGHTS_1D,
    HISTOGRAM_BINS,
    compute_equalize_lut,
)

GRAYSCALE_R = 0.299
GRAYSCALE_G = 0.587
GRAYSCALE_B = 0.114
EDGE_MARGIN = 1
SOBEL_WEIGHT = 2.0

@njit(parallel=True)
def _to_grayscale_jit(image: np.ndarray, output: np.ndarray) -> None:
    rows, cols = image.shape[:2]
    for x in prange(rows):
        for y in range(cols):
            r = image[x, y, 0]
            g = image[x, y, 1]
            b = image[x, y, 2]
            output[x, y] = GRAYSCALE_R * r + GRAYSCALE_G * g + GRAYSCALE_B * b

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte la imagen a escala de grises manualmente."""
    output = np.empty(image.shape[:2], dtype=np.uint8)
    _to_grayscale_jit(image, output)
    return output

@njit(parallel=True)
def _detect_edges_jit(image: np.ndarray, output: np.ndarray) -> None:
    rows, cols = image.shape
    for x in prange(EDGE_MARGIN, rows - EDGE_MARGIN):
        for y in range(EDGE_MARGIN, cols - EDGE_MARGIN):
            gx = (
                -float(image[x-EDGE_MARGIN, y-EDGE_MARGIN]) + float(image[x-EDGE_MARGIN, y+EDGE_MARGIN])
                - SOBEL_WEIGHT*float(image[x, y-EDGE_MARGIN]) + SOBEL_WEIGHT*float(image[x, y+EDGE_MARGIN])
                - float(image[x+EDGE_MARGIN, y-EDGE_MARGIN]) + float(image[x+EDGE_MARGIN, y+EDGE_MARGIN])
            )
            gy = (
                -float(image[x-EDGE_MARGIN, y-EDGE_MARGIN]) - SOBEL_WEIGHT*float(image[x-EDGE_MARGIN, y]) - float(image[x-EDGE_MARGIN, y+EDGE_MARGIN])
                + float(image[x+EDGE_MARGIN, y-EDGE_MARGIN]) + SOBEL_WEIGHT*float(image[x+EDGE_MARGIN, y]) + float(image[x+EDGE_MARGIN, y+EDGE_MARGIN])
            )
            magnitude = math.sqrt(gx*gx + gy*gy)
            output[x, y] = min(magnitude, float(MAX_PIXEL_VALUE))

def _detect_edges(image: np.ndarray) -> np.ndarray:
    """Detecta bordes con filtro Sobel manual 2D."""
    gray = _to_grayscale(image)
    output = np.zeros_like(gray)
    _detect_edges_jit(gray, output)
    return output

def _gaussian_mask() -> np.ndarray:
    weights = np.array(GAUSSIAN_BLUR_WEIGHTS_1D, dtype=np.float32)
    return np.outer(weights, weights)

@njit(parallel=True)
def _apply_blur_jit(image: np.ndarray, output: np.ndarray, mask: np.ndarray, radius: int) -> None:
    rows, cols, channels = image.shape
    for x in prange(rows):
        for y in range(cols):
            for c in range(channels):
                acc = 0.0
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        px = min(max(x + i, 0), rows - 1)
                        py = min(max(y + j, 0), cols - 1)
                        acc += mask[i + radius, j + radius] * image[px, py, c]
                output[x, y, c] = acc

def _apply_blur(image: np.ndarray) -> np.ndarray:
    """Aplica desenfoque gaussiano iterando sobre ventana 2D."""
    output = np.zeros_like(image)
    mask = _gaussian_mask()
    _apply_blur_jit(image, output, mask, BLUR_RADIUS)
    return output

@njit(parallel=True)
def _map_lut_jit(gray: np.ndarray, lut: np.ndarray, output: np.ndarray) -> None:
    rows, cols = gray.shape
    for x in prange(rows):
        for y in range(cols):
            output[x, y] = lut[gray[x, y]]

def _equalize(image: np.ndarray) -> np.ndarray:
    """Ecualiza el histograma usando bincount y mapeo manual."""
    gray = _to_grayscale(image)
    hist = np.bincount(gray.ravel(), minlength=HISTOGRAM_BINS).astype(np.uint32)
    lut = compute_equalize_lut(hist)
    output = np.empty_like(gray)
    _map_lut_jit(gray, lut, output)
    return output

_OPERATIONS = {
    'grayscale': _to_grayscale,
    'edges': _detect_edges,
    'blur': _apply_blur,
    'equalize': _equalize,
}

class CPUBackend(GPUBackend):
    """Aplica operaciones de transformación de imágenes en CPU."""

    def __init__(self):
        self.backend_name = CPU_BACKEND_NAME
        self.device_info = "CPU"

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        if operation not in VALID_OPERATIONS:
            raise ValueError(f'Unknown operation: {operation!r}')
        return _OPERATIONS[operation](image)

    def warmup(self, operation: str) -> None:
        shape = (32, 32, 3)
        dummy = np.zeros(shape, dtype=np.uint8)
        self.process(dummy, operation)

