"""Implementación del backend de procesamiento para OpenCL."""

# Los kernels OpenCL (grayscale/edges/blur/histogram/map) extienden el módulo.
# pylint: disable=too-many-lines
# El contrato de fallback OOM (process/_handle_oom) es idéntico al de CUDA
# por diseño: cada backend Strategy lo implementa con su propio _GPU_OOM_ERRORS.
# pylint: disable=duplicate-code

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from core.backend.base import (
    GPUBackend,
    OPENCL_BACKEND_NAME,
    NO_BATCH_TIME,
    _gpu_semaphore,
    MAX_PIXEL_VALUE,
    VALID_OPERATIONS,
    BLUR_KERNEL_SIZE,
    BLUR_RADIUS,
    GAUSSIAN_BLUR_WEIGHTS_1D,
    HISTOGRAM_BINS,
    compute_equalize_lut,
)

_MS_PER_SECOND: float = 1000.0
from core.backend.cpu import CPUBackend, _to_grayscale
from core.backend.cuda import WARMUP_IMAGE_SIZE

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

    __kernel void grayscale_batch(
        __global const unsigned char* images,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int n = get_global_id(0);
        int x = get_global_id(1);
        int y = get_global_id(2);

        if (x < rows && y < cols) {{
            int in_idx = ((n * rows + x) * cols + y) * {OPENCL_IMAGE_CHANNELS};
            int out_idx = (n * rows + x) * cols + y;

            float r = (float)images[in_idx];
            float g = (float)images[in_idx + 1];
            float b = (float)images[in_idx + 2];

            output[out_idx] = (unsigned char)(
                {OPENCL_GRAYSCALE_RED_WEIGHT}f * r +
                {OPENCL_GRAYSCALE_GREEN_WEIGHT}f * g +
                {OPENCL_GRAYSCALE_BLUE_WEIGHT}f * b
            );
        }}
    }}

    __kernel void edges_batch(
        __global const unsigned char* grays,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int n = get_global_id(0);
        int x = get_global_id(1);
        int y = get_global_id(2);

        if (x >= {OPENCL_EDGE_MARGIN} && x < rows - {OPENCL_EDGE_MARGIN} && y >= {OPENCL_EDGE_MARGIN} && y < cols - {OPENCL_EDGE_MARGIN}) {{
            int base = n * rows * cols;
            float gx =
                -(float)grays[base + (x-{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] + (float)grays[base + (x-{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})]
                -{OPENCL_SOBEL_WEIGHT}f * (float)grays[base + x*cols + (y-{OPENCL_EDGE_MARGIN})] + {OPENCL_SOBEL_WEIGHT}f * (float)grays[base + x*cols + (y+{OPENCL_EDGE_MARGIN})]
                -(float)grays[base + (x+{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] + (float)grays[base + (x+{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})];

            float gy =
                -(float)grays[base + (x-{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] - {OPENCL_SOBEL_WEIGHT}f * (float)grays[base + (x-{OPENCL_EDGE_MARGIN})*cols + y] - (float)grays[base + (x-{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})]
                +(float)grays[base + (x+{OPENCL_EDGE_MARGIN})*cols + (y-{OPENCL_EDGE_MARGIN})] + {OPENCL_SOBEL_WEIGHT}f * (float)grays[base + (x+{OPENCL_EDGE_MARGIN})*cols + y] + (float)grays[base + (x+{OPENCL_EDGE_MARGIN})*cols + (y+{OPENCL_EDGE_MARGIN})];

            float magnitude = sqrt(gx*gx + gy*gy);
            if (magnitude > (float){MAX_PIXEL_VALUE}) {{
                magnitude = (float){MAX_PIXEL_VALUE};
            }}
            output[base + x * cols + y] = (unsigned char)magnitude;
        }}
    }}

    __kernel void blur_batch(
        __global const unsigned char* images,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int n = get_global_id(0);
        int x = get_global_id(1);
        int y = get_global_id(2);
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
                        acc += w * (float)images[
                            ((n * rows + px) * cols + py) * channels + c];
                    }}
                }}
                output[((n * rows + x) * cols + y) * channels + c] =
                    (unsigned char)acc;
            }}
        }}
    }}

    __kernel void histogram_batch(
        __global const unsigned char* grays,
        __global uint* hist,
        const int rows,
        const int cols)
    {{
        int n = get_global_id(0);
        int x = get_global_id(1);
        int y = get_global_id(2);

        if (x < rows && y < cols) {{
            int idx = (n * rows + x) * cols + y;
            atomic_add(&hist[n * {HISTOGRAM_BINS} + grays[idx]], 1);
        }}
    }}

    __kernel void map_lut_batch(
        __global const unsigned char* grays,
        __global const unsigned char* luts,
        __global unsigned char* output,
        const int rows,
        const int cols)
    {{
        int n = get_global_id(0);
        int x = get_global_id(1);
        int y = get_global_id(2);

        if (x < rows && y < cols) {{
            int idx = (n * rows + x) * cols + y;
            output[idx] = luts[n * {HISTOGRAM_BINS} + grays[idx]];
        }}
    }}
    """
    # pylint: enable=line-too-long

    _OPERATIONS_OPENCL = {
        'grayscale': 'grayscale',
        'edges': 'edges',
    }
    _OPERATIONS_OPENCL_BATCH = {
        'grayscale': 'grayscale_batch',
        'edges': 'edges_batch',
    }
    _KERNEL_NAMES: tuple[str, ...] = (
        'grayscale', 'edges', 'blur', 'histogram', 'map_lut',
        'grayscale_batch', 'edges_batch', 'blur_batch',
        'histogram_batch', 'map_lut_batch',
    )
    _GPU_OOM_ERRORS: tuple[type[Exception], ...] = (
        MemoryError, cl.MemoryError)
else:
    _OPENCL_KERNELS_SOURCE = ""
    _OPERATIONS_OPENCL = {}
    _OPERATIONS_OPENCL_BATCH = {}
    _KERNEL_NAMES = ()
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
        self._kernels = {
            name: cl.Kernel(self.program, name) for name in _KERNEL_NAMES
        }

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

    def warmup(self, operation: str) -> None:
        """Ejercita y calienta el camino simple y el camino batch.

        Args:
            operation: Transformación a ejercitar. Valores: VALID_OPERATIONS.
        """
        shape = (WARMUP_IMAGE_SIZE, WARMUP_IMAGE_SIZE, OPENCL_IMAGE_CHANNELS)
        dummy = np.zeros(shape, dtype=np.uint8)
        self.process(dummy, operation)
        self.process_batch([dummy, dummy], operation)

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
        kernel_func = self._kernels[_OPERATIONS_OPENCL[operation]]
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
        self._kernels['blur'](
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

    def process_batch(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> tuple[list[np.ndarray], float | None]:
        """Procesa el lote en una sola transferencia PCIe H2D y D2H.

        Args:
            images: Lista de arrays del mismo shape, dtype uint8.
            operation: Transformación a aplicar. Valores: VALID_OPERATIONS.

        Returns:
            Tupla (resultados, per_image_ms) con tiempo amortizado por imagen.
        """
        with _gpu_semaphore:
            try:
                start = time.perf_counter()
                results = self._dispatch_batch_cl(images, operation)
                ms = (time.perf_counter() - start) * _MS_PER_SECOND
                return results, ms / len(images)
            except _GPU_OOM_ERRORS as exc:
                self._handle_oom(images[0], exc)
                fallback = [self._cpu_fallback.process(img, operation)
                            for img in images]
                return fallback, NO_BATCH_TIME

    def _dispatch_batch_cl(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> list[np.ndarray]:
        """Enruta el lote al helper correspondiente a la operación."""
        if operation == 'blur':
            return self._batch_blur_cl(images)
        if operation == 'equalize':
            return self._batch_equalize_cl(images)
        return self._batch_simple_cl(images, operation)

    def _make_batch_buffers(
        self,
        stacked_in: np.ndarray,
        out_nbytes: int,
    ) -> tuple:
        """Crea un buffer READ con el lote y un buffer WRITE para la salida."""
        mf = cl.mem_flags
        buf_in = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=stacked_in)
        buf_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, out_nbytes)
        return buf_in, buf_out

    def _batch_global_size(
        self, shape: tuple[int, int], n: int
    ) -> tuple[int, int, int]:
        """Calcula el global work size (N, H, W) de un kernel batcheado."""
        rows, cols = shape
        return (n, rows, cols)

    def _batch_simple_cl(
        self,
        images: list[np.ndarray],
        operation: str,
    ) -> list[np.ndarray]:
        """Ejecuta grayscale/edges en lote con un solo kernel 3D."""
        imgs = ([_to_grayscale(img) for img in images]
                if operation == 'edges' else images)
        n, h, w = len(imgs), imgs[0].shape[0], imgs[0].shape[1]
        stacked_in = np.ascontiguousarray(np.stack(imgs))
        host_out = np.zeros((n, h, w), dtype=np.uint8)
        buf_in, buf_out = self._make_batch_buffers(stacked_in, host_out.nbytes)
        kernel_fn = self._kernels[_OPERATIONS_OPENCL_BATCH[operation]]
        kernel_fn(
            self.queue, self._batch_global_size((h, w), n), None,
            buf_in, buf_out, np.int32(h), np.int32(w))
        cl.enqueue_copy(self.queue, host_out, buf_out).wait()
        return [host_out[i] for i in range(n)]

    def _batch_blur_cl(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Aplica desenfoque gaussiano en lote con un solo kernel 3D."""
        stacked_in = np.ascontiguousarray(np.stack(images))
        n, h, w, _ = stacked_in.shape
        host_out = np.zeros_like(stacked_in)
        buf_in, buf_out = self._make_batch_buffers(stacked_in, host_out.nbytes)
        self._kernels['blur_batch'](
            self.queue, self._batch_global_size((h, w), n), None,
            buf_in, buf_out, np.int32(h), np.int32(w))
        cl.enqueue_copy(self.queue, host_out, buf_out).wait()
        return [host_out[i] for i in range(n)]

    def _compute_batch_histograms_cl(
        self,
        stacked_g: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Calcula los N histogramas del lote con un solo kernel 3D."""
        h, w = stacked_g.shape[1], stacked_g.shape[2]
        histograms = np.zeros((n, HISTOGRAM_BINS), dtype=np.uint32)
        buf_g = self._read_buffer(stacked_g)
        buf_h = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=histograms)
        self._kernels['histogram_batch'](
            self.queue, self._batch_global_size((h, w), n), None,
            buf_g, buf_h, np.int32(h), np.int32(w))
        cl.enqueue_copy(self.queue, histograms, buf_h).wait()
        return histograms

    def _compute_batch_luts_cl(
        self,
        stacked_g: np.ndarray,
        n: int,
    ) -> list[np.ndarray]:
        """Calcula la LUT de ecualización de cada imagen del lote."""
        histograms = self._compute_batch_histograms_cl(stacked_g, n)
        return [compute_equalize_lut(histograms[i]) for i in range(n)]

    def _batch_equalize_cl(
        self,
        images: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Ecualiza histograma en lote; un solo kernel 3D por etapa."""
        grays = [_to_grayscale(img) for img in images]
        stacked_g = np.ascontiguousarray(np.stack(grays))
        n, h, w = stacked_g.shape
        luts = self._compute_batch_luts_cl(stacked_g, n)
        stacked_l = np.ascontiguousarray(np.stack(luts))
        host_out = np.zeros((n, h, w), dtype=np.uint8)
        buf_g, buf_o = self._make_batch_buffers(stacked_g, host_out.nbytes)
        buf_l = self._read_buffer(stacked_l)
        self._kernels['map_lut_batch'](
            self.queue, self._batch_global_size((h, w), n), None,
            buf_g, buf_l, buf_o, np.int32(h), np.int32(w))
        cl.enqueue_copy(self.queue, host_out, buf_o).wait()
        return [host_out[i] for i in range(n)]

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
        self._kernels['histogram'](
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
        self._kernels['map_lut'](
            self.queue, (rows, cols), None,
            gray_buf, lut_buf, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output
