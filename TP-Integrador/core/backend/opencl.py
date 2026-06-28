"""Implementación del backend de procesamiento para OpenCL."""

# Los kernels OpenCL (grayscale/edges/blur/histogram/map) extienden el módulo.
# pylint: disable=too-many-lines
# El contrato de fallback OOM (process/_handle_oom) es idéntico al de CUDA
# por diseño: cada backend Strategy lo implementa con su propio _GPU_OOM_ERRORS.
# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from core.backend.base import (
    GPUBackend,
    OPENCL_BACKEND_NAME,
    _gpu_semaphore,
    MAX_PIXEL_VALUE,
    VALID_OPERATIONS,
    BLUR_KERNEL_SIZE,
    BLUR_RADIUS,
    GAUSSIAN_BLUR_WEIGHTS_1D,
    HISTOGRAM_BINS,
    compute_equalize_lut,
)
from core.backend.cpu import CPUBackend, _to_grayscale

try:
    import pyopencl as cl
    _OPENCL_AVAILABLE = True
except ImportError:
    cl = None
    _OPENCL_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

OPENCL_GRAYSCALE_RED_WEIGHT: float = 0.299
OPENCL_GRAYSCALE_GREEN_WEIGHT: float = 0.587
OPENCL_GRAYSCALE_BLUE_WEIGHT: float = 0.114
OPENCL_IMAGE_CHANNELS: int = 3
OPENCL_EDGE_MARGIN: int = 1
OPENCL_SOBEL_WEIGHT: float = 2.0
OPENCL_BLUR_RADIUS: int = BLUR_RADIUS
_BLUR_WEIGHTS_C: str = ', '.join(f'{w}f' for w in GAUSSIAN_BLUR_WEIGHTS_1D)

if _OPENCL_AVAILABLE:
    # pylint: disable=line-too-long
    _OPENCL_KERNELS_SOURCE = f"""
    __kernel void grayscale(
        __global const unsigned char* image,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x < rows && y < cols) {{
            int in_idx = (x * cols + y) * {OPENCL_IMAGE_CHANNELS};
            int out_idx = x * cols + y;

            float r = (float)image[in_idx];
            float g = (float)image[in_idx + 1];
            float b = (float)image[in_idx + 2];

            output[out_idx] = (unsigned char)(
                {OPENCL_GRAYSCALE_RED_WEIGHT}f * r +
                {OPENCL_GRAYSCALE_GREEN_WEIGHT}f * g +
                {OPENCL_GRAYSCALE_BLUE_WEIGHT}f * b
            );
        }}
    }}

    __kernel void edges(
        __global const unsigned char* image,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x >= {OPENCL_EDGE_MARGIN} && x < rows - {OPENCL_EDGE_MARGIN} && y >= {OPENCL_EDGE_MARGIN} && y < cols - {OPENCL_EDGE_MARGIN}) {{
            float gx =
                -(float)image[(x-{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] + (float)image[(x-{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})]
                -{OPENCL_SOBEL_WEIGHT}f * (float)image[x*cols + (y-{OPENCL_EDGE_MARGIN})] + {OPENCL_SOBEL_WEIGHT}f * (float)image[x*cols + (y+{OPENCL_EDGE_MARGIN})]
                -(float)image[(x+{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] + (float)image[(x+{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})];

            float gy =
                -(float)image[(x-{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] - {OPENCL_SOBEL_WEIGHT}f * (float)image[(x-{OPENCL_EDGE_MARGIN})*cols + y] - (float)image[(x-{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})]
                +(float)image[(x+{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] + {OPENCL_SOBEL_WEIGHT}f * (float)image[(x+{OPENCL_EDGE_MARGIN})*cols + y] + (float)image[(x+{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})];

            float magnitude = sqrt(gx*gx + gy*gy);
            if (magnitude > (float){MAX_PIXEL_VALUE}) {{
                magnitude = (float){MAX_PIXEL_VALUE};
            }}
            output[x * cols + y] = (unsigned char)magnitude;
        }}
    }}

    __kernel void blur(
        __global const unsigned char* image,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int x = get_global_id(0);
        int y = get_global_id(1);
        const int radius = {OPENCL_BLUR_RADIUS};
        const int channels = {OPENCL_IMAGE_CHANNELS};
        float weights[{BLUR_KERNEL_SIZE}] = {{{_BLUR_WEIGHTS_C}}};

        if (x < rows && y < cols) {{
            for (int c = 0; c < channels; c++) {{
                float acc = 0.0f;
                for (int i = -radius; i <= radius; i++) {{
                    for (int j = -radius; j <= radius; j++) {{
                        int px = min(max(x + i, 0), rows - 1);
                        int py = min(max(y + j, 0), cols - 1);
                        float w = weights[i + radius] * weights[j + radius];
                        acc += w * (float)image[(px * cols + py) * channels + c];
                    }}
                }}
                output[(x * cols + y) * channels + c] = (unsigned char)acc;
            }}
        }}
    }}

    __kernel void histogram(
        __global const unsigned char* gray,
        __global uint* hist,
        const int rows,
        const int cols)
    {{
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x < rows && y < cols) {{
            atomic_add(&hist[gray[x * cols + y]], 1);
        }}
    }}

    __kernel void map_lut(
        __global const unsigned char* gray,
        __global const unsigned char* lut,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x < rows && y < cols) {{
            output[x * cols + y] = lut[gray[x * cols + y]];
        }}
    }}
    """
    # pylint: enable=line-too-long

    _OPERATIONS_OPENCL = {
        'grayscale': 'grayscale',
        'edges': 'edges',
    }
    _GPU_OOM_ERRORS: tuple[type[Exception], ...] = (
        MemoryError, cl.MemoryError)
else:
    _OPENCL_KERNELS_SOURCE = ""
    _OPERATIONS_OPENCL = {}
    _GPU_OOM_ERRORS = (MemoryError,)


class OpenCLBackend(GPUBackend):
    """Procesa imágenes en GPU AMD/Intel mediante kernels OpenCL."""

    def __init__(self, on_fallback: Callable[[], None] | None = None):
        if not _OPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL no está disponible")

        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No se encontraron plataformas OpenCL")

        device = platforms[0].get_devices()[0]
        self.backend_name = OPENCL_BACKEND_NAME
        self.device_info = device.name
        self._on_fallback = on_fallback
        self._cpu_fallback = CPUBackend()
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, _OPENCL_KERNELS_SOURCE).build()

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Aplica la operación a la imagen usando un kernel OpenCL.

        Args:
            image: Array NumPy con shape (H, W, C), dtype uint8.
            operation: Transformación a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Array procesado; cae a CPUBackend ante un OOM de GPU.

        Raises:
            ValueError: Si operation no está en VALID_OPERATIONS.
        """
        if operation not in VALID_OPERATIONS:
            raise ValueError(f'Unknown operation: {operation!r}')
        with _gpu_semaphore:
            try:
                return self._process_on_gpu(image, operation)
            except _GPU_OOM_ERRORS as exc:
                self._handle_oom(image, exc)
                return self._cpu_fallback.process(image, operation)

    def _handle_oom(self, image: np.ndarray, exc: Exception) -> None:
        """Loggea el OOM de GPU y notifica el fallback configurado."""
        _LOGGER.warning(
            'GPU OOM (%s), fallback a CPU: %r', image.shape, exc)
        if self._on_fallback is not None:
            self._on_fallback()

    def _process_on_gpu(
        self, image: np.ndarray, operation: str) -> np.ndarray:
        """Enruta la operación al conjunto de kernels correspondiente."""
        if operation == 'blur':
            return self._blur(image)
        if operation == 'equalize':
            return self._equalize(image)
        return self._run_simple(image, operation)

    def _create_buffers(self, image: np.ndarray, output: np.ndarray) -> tuple:
        """Crea los buffers de lectura y escritura para OpenCL."""
        mf = cl.mem_flags
        image_contig = np.ascontiguousarray(image)
        img_buf = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_contig
        )
        out_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes)
        return img_buf, out_buf

    def _run_simple(self, image: np.ndarray, operation: str) -> np.ndarray:
        """Ejecuta los kernels de un solo paso (grayscale/edges)."""
        if operation == "edges":
            image = _to_grayscale(image)
        rows, cols = image.shape[:2]
        output = np.zeros((rows, cols), dtype=np.uint8)
        img_buf, out_buf = self._create_buffers(image, output)
        kernel_func = getattr(self.program, _OPERATIONS_OPENCL[operation])
        kernel_func(
            self.queue, (rows, cols), None,
            img_buf, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

    def _blur(self, image: np.ndarray) -> np.ndarray:
        """Aplica el desenfoque gaussiano conservando los 3 canales."""
        rows, cols = image.shape[:2]
        output = np.zeros(
            (rows, cols, OPENCL_IMAGE_CHANNELS), dtype=np.uint8)
        img_buf, out_buf = self._create_buffers(image, output)
        self.program.blur(
            self.queue, (rows, cols), None,
            img_buf, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

    def _equalize(self, image: np.ndarray) -> np.ndarray:
        """Ecualiza el histograma sobre la imagen en escala de grises."""
        gray = _to_grayscale(image)
        histogram = self._compute_histogram(gray)
        lut = compute_equalize_lut(histogram)
        return self._apply_lut(gray, lut)

    def _read_buffer(self, host: np.ndarray):
        """Crea un buffer de solo lectura copiando ``host`` a la GPU."""
        mf = cl.mem_flags
        return cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=np.ascontiguousarray(host))

    def _compute_histogram(self, gray: np.ndarray) -> np.ndarray:
        """Calcula el histograma del gris en GPU con sumas atómicas."""
        rows, cols = gray.shape[:2]
        histogram = np.zeros(HISTOGRAM_BINS, dtype=np.uint32)
        gray_buf = self._read_buffer(gray)
        hist_buf = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=histogram)
        self.program.histogram(
            self.queue, (rows, cols), None,
            gray_buf, hist_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, histogram, hist_buf).wait()
        return histogram

    def _apply_lut(self, gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Mapea cada píxel del gris a través de la LUT en GPU."""
        rows, cols = gray.shape[:2]
        output = np.zeros((rows, cols), dtype=np.uint8)
        gray_buf = self._read_buffer(gray)
        lut_buf = self._read_buffer(lut)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        self.program.map_lut(
            self.queue, (rows, cols), None,
            gray_buf, lut_buf, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output
