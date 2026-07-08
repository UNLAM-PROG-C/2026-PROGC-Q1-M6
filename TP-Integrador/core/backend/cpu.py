"""Implementación del backend de procesamiento para CPU (Numba JIT)."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from core.backend.base import (
    CPU_BACKEND_NAME,
    GPUBackend,
    MAX_PIXEL_VALUE,
    NO_BATCH_TIME,
    VALID_OPERATIONS,
    BLUR_RADIUS,
    GAUSSIAN_BLUR_WEIGHTS_1D,
    HISTOGRAM_BINS,
    compute_equalize_lut,
    OP_GRAYSCALE,
    OP_EDGES,
    OP_BLUR,
    OP_EQUALIZE,
    GRAYSCALE_R,
    GRAYSCALE_G,
    GRAYSCALE_B,
    EDGE_MARGIN,
    SOBEL_WEIGHT,
)




@njit(parallel=True, fastmath=True)
def grayscale_kernel(image: np.ndarray, output: np.ndarray, rows: int, cols: int) -> None:
    for x in prange(rows):
        for y in range(cols):
            r = image[x, y, 0]
            g = image[x, y, 1]
            b = image[x, y, 2]
            output[x, y] = GRAYSCALE_R * r + GRAYSCALE_G * g + GRAYSCALE_B * b


@njit(parallel=True, fastmath=True)
def edges_kernel(gray: np.ndarray, output: np.ndarray, rows: int, cols: int) -> None:
    for x in prange(EDGE_MARGIN, rows - EDGE_MARGIN):
        for y in range(EDGE_MARGIN, cols - EDGE_MARGIN):
            gx = (
                -float(gray[x-1, y-1]) + float(gray[x-1, y+1])
                - SOBEL_WEIGHT*float(gray[x, y-1]) + SOBEL_WEIGHT*float(gray[x, y+1])
                - float(gray[x+1, y-1]) + float(gray[x+1, y+1])
            )
            gy = (
                -float(gray[x-1, y-1]) - SOBEL_WEIGHT*float(gray[x-1, y]) - float(gray[x-1, y+1])
                + float(gray[x+1, y-1]) + SOBEL_WEIGHT*float(gray[x+1, y]) + float(gray[x+1, y+1])
            )
            mag = math.sqrt(gx*gx + gy*gy)
            output[x, y] = min(mag, MAX_PIXEL_VALUE)


@njit(parallel=True, fastmath=True)
def blur_kernel(image: np.ndarray, output: np.ndarray, mask: np.ndarray,
                radius: int, rows: int, cols: int, channels: int) -> None:
    for x in prange(rows):
        for y in range(cols):
            for c in range(channels):
                acc = 0.0
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        px = min(max(x + i, 0), rows - 1)
                        py = min(max(y + j, 0), cols - 1)
                        w = mask[i + radius, j + radius]
                        acc += w * image[px, py, c]
                output[x, y, c] = acc


@njit(parallel=True, fastmath=True)
def map_lut_kernel(gray: np.ndarray, lut: np.ndarray, output: np.ndarray,
                   rows: int, cols: int) -> None:
    for x in prange(rows):
        for y in range(cols):
            output[x, y] = lut[gray[x, y]]


class CPUBackend(GPUBackend):
    """Procesa imágenes en CPU usando kernels JIT de Numba."""

    def __init__(self):
        self.backend_name = CPU_BACKEND_NAME
        self.device_info = "Numba JIT"

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        if operation not in VALID_OPERATIONS:
            raise ValueError(f'Unknown operation: {operation!r}')
        if operation == OP_GRAYSCALE:
            return self._run_grayscale(image)
        if operation == OP_EDGES:
            return self._run_edges(image)
        if operation == OP_BLUR:
            return self._run_blur(image)
        if operation == OP_EQUALIZE:
            return self._run_equalize(image)
        raise ValueError(f'Unknown operation: {operation!r}')

    def process_batch(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> tuple[list[np.ndarray], float | None]:
        results = [self.process(img, operation) for img in images]
        return results, NO_BATCH_TIME

    def warmup(self, operation: str) -> None:
        shape = (32, 32, 3)
        dummy = np.zeros(shape, dtype=np.uint8)
        self.process(dummy, operation)

    # -------------------------------------------------------------------------
    # Single Dispatchers
    # -------------------------------------------------------------------------
    def _run_grayscale(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        output = np.empty((rows, cols), dtype=np.uint8)
        grayscale_kernel(image, output, rows, cols)
        return output

    def _run_edges(self, image: np.ndarray) -> np.ndarray:
        gray = self._run_grayscale(image)
        rows, cols = gray.shape[:2]
        output = np.zeros_like(gray)
        edges_kernel(gray, output, rows, cols)
        return output

    def _run_blur(self, image: np.ndarray) -> np.ndarray:
        rows, cols, channels = image.shape
        output = np.zeros_like(image)
        mask = _gaussian_mask()
        blur_kernel(image, output, mask, BLUR_RADIUS, rows, cols, channels)
        return output

    def _run_equalize(self, image: np.ndarray) -> np.ndarray:
        gray = self._run_grayscale(image)
        rows, cols = gray.shape[:2]
        hist = np.bincount(gray.ravel(), minlength=HISTOGRAM_BINS).astype(np.uint32)
        lut = compute_equalize_lut(hist)
        output = np.empty_like(gray)
        map_lut_kernel(gray, lut, output, rows, cols)
        return output


def _gaussian_mask() -> np.ndarray:
    weights = np.array(GAUSSIAN_BLUR_WEIGHTS_1D, dtype=np.float32)
    return np.outer(weights, weights)
