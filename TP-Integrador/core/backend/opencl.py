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
    _gpu_semaphore,
    OP_GRAYSCALE,
    OP_EDGES,
    OP_BLUR,
    OP_EQUALIZE,
    GRAYSCALE_R,
    GRAYSCALE_G,
    GRAYSCALE_B,
    EDGE_MARGIN,
    SOBEL_WEIGHT,
    WARMUP_IMAGE_SIZE,
    GPU_THREADS_PER_BLOCK,
    RGB_CHANNELS,
)

from core.backend.cpu import CPUBackend

try:
    import pyopencl as cl
    _OPENCL_AVAILABLE = True
except ImportError:
    cl = None
    _OPENCL_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

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
            int in_idx = (x * cols + y) * {RGB_CHANNELS};
            int out_idx = x * cols + y;

            float r = (float)image[in_idx];
            float g = (float)image[in_idx + 1];
            float b = (float)image[in_idx + 2];

            output[out_idx] = (unsigned char)(
                {GRAYSCALE_R}f * r +
                {GRAYSCALE_G}f * g +
                {GRAYSCALE_B}f * b
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

        if (x >= {EDGE_MARGIN} && x < rows - {EDGE_MARGIN} && y >= {EDGE_MARGIN} && y < cols - {EDGE_MARGIN}) {{
            float gx =
                -(float)image[(x-{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] + (float)image[(x-{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})]
                -{SOBEL_WEIGHT}f * (float)image[x*cols + (y-{EDGE_MARGIN})] + {SOBEL_WEIGHT}f * (float)image[x*cols + (y+{EDGE_MARGIN})]
                -(float)image[(x+{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] + (float)image[(x+{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})];

            float gy =
                -(float)image[(x-{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] - {SOBEL_WEIGHT}f * (float)image[(x-{EDGE_MARGIN})*cols + y] - (float)image[(x-{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})]
                +(float)image[(x+{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] + {SOBEL_WEIGHT}f * (float)image[(x+{EDGE_MARGIN})*cols + y] + (float)image[(x+{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})];

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
        const int radius = {BLUR_RADIUS};
        const int channels = {RGB_CHANNELS};
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
            int in_idx = ((n * rows + x) * cols + y) * {RGB_CHANNELS};
            int out_idx = (n * rows + x) * cols + y;

            float r = (float)images[in_idx];
            float g = (float)images[in_idx + 1];
            float b = (float)images[in_idx + 2];

            output[out_idx] = (unsigned char)(
                {GRAYSCALE_R}f * r +
                {GRAYSCALE_G}f * g +
                {GRAYSCALE_B}f * b
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

        if (x >= {EDGE_MARGIN} && x < rows - {EDGE_MARGIN} && y >= {EDGE_MARGIN} && y < cols - {EDGE_MARGIN}) {{
            int base = n * rows * cols;
            float gx =
                -(float)grays[base + (x-{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] + (float)grays[base + (x-{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})]
                -{SOBEL_WEIGHT}f * (float)grays[base + x*cols + (y-{EDGE_MARGIN})] + {SOBEL_WEIGHT}f * (float)grays[base + x*cols + (y+{EDGE_MARGIN})]
                -(float)grays[base + (x+{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] + (float)grays[base + (x+{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})];

            float gy =
                -(float)grays[base + (x-{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] - {SOBEL_WEIGHT}f * (float)grays[base + (x-{EDGE_MARGIN})*cols + y] - (float)grays[base + (x-{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})]
                +(float)grays[base + (x+{EDGE_MARGIN})*cols + (y-{EDGE_MARGIN})] + {SOBEL_WEIGHT}f * (float)grays[base + (x+{EDGE_MARGIN})*cols + y] + (float)grays[base + (x+{EDGE_MARGIN})*cols + (y+{EDGE_MARGIN})];

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
        const int radius = {BLUR_RADIUS};
        const int channels = {RGB_CHANNELS};
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

    _KERNEL_NAMES: tuple[str, ...] = (
        'grayscale', 'edges', 'blur', 'histogram', 'map_lut',
        'grayscale_batch', 'edges_batch', 'blur_batch',
        'histogram_batch', 'map_lut_batch',
    )
    _GPU_OOM_ERRORS: tuple[type[Exception], ...] = (
        MemoryError, cl.MemoryError)
else:
    _OPENCL_KERNELS_SOURCE = ""

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

        device = None
        for p in platforms:
            for d in p.get_devices():
                device = d
                break
            if device:
                break
                
        if not device:
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

    def _get_work_sizes(self, shape: tuple) -> tuple:
        local_size = (GPU_THREADS_PER_BLOCK, GPU_THREADS_PER_BLOCK)
        global_size = (
            ((shape[0] + (GPU_THREADS_PER_BLOCK - 1)) // GPU_THREADS_PER_BLOCK) * GPU_THREADS_PER_BLOCK,
            ((shape[1] + (GPU_THREADS_PER_BLOCK - 1)) // GPU_THREADS_PER_BLOCK) * GPU_THREADS_PER_BLOCK
        )
        return global_size, local_size

    def _get_batch_work_sizes(self, shape: tuple, n: int) -> tuple:
        local_size = (1, GPU_THREADS_PER_BLOCK, GPU_THREADS_PER_BLOCK)
        global_size = (
            n,
            ((shape[0] + (GPU_THREADS_PER_BLOCK - 1)) // GPU_THREADS_PER_BLOCK) * GPU_THREADS_PER_BLOCK,
            ((shape[1] + (GPU_THREADS_PER_BLOCK - 1)) // GPU_THREADS_PER_BLOCK) * GPU_THREADS_PER_BLOCK
        )
        return global_size, local_size

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
        shape = (WARMUP_IMAGE_SIZE, WARMUP_IMAGE_SIZE, RGB_CHANNELS)
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
        if operation == OP_GRAYSCALE:
            return self._run_grayscale(image)
        if operation == OP_EDGES:
            return self._run_edges(image)
        if operation == OP_BLUR:
            return self._run_blur(image)
        if operation == OP_EQUALIZE:
            return self._run_equalize(image)
        raise ValueError(f'Unknown operation: {operation!r}')

    def _create_buffers(self, image: np.ndarray, output: np.ndarray) -> tuple:
        mf = cl.mem_flags
        image_contig = np.ascontiguousarray(image)
        img_buf = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_contig
        )
        out_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes)
        return img_buf, out_buf

    def _run_grayscale(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        output = np.zeros((rows, cols), dtype=np.uint8)
        img_buf, out_buf = self._create_buffers(image, output)
        g_size, l_size = self._get_work_sizes((rows, cols))
        self._kernels['grayscale'](
            self.queue, g_size, l_size,
            img_buf, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

    def _run_edges(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        output = np.zeros((rows, cols), dtype=np.uint8)
        gray_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, rows * cols)
        
        # d_image buffer
        img_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(image))
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        g_size, l_size = self._get_work_sizes((rows, cols))
        
        self._kernels['grayscale'](
            self.queue, g_size, l_size,
            img_buf, gray_buf, np.int32(rows), np.int32(cols))
            
        self._kernels['edges'](
            self.queue, g_size, l_size,
            gray_buf, out_buf, np.int32(rows), np.int32(cols))
            
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

    def _run_blur(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        output = np.zeros(
            (rows, cols, RGB_CHANNELS), dtype=np.uint8)
        img_buf, out_buf = self._create_buffers(image, output)
        g_size, l_size = self._get_work_sizes((rows, cols))
        self._kernels['blur'](
            self.queue, g_size, l_size,
            img_buf, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

    def _run_equalize(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape[:2]
        gray_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, rows * cols)
        img_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(image))
        
        g_size, l_size = self._get_work_sizes((rows, cols))
        
        self._kernels['grayscale'](
            self.queue, g_size, l_size,
            img_buf, gray_buf, np.int32(rows), np.int32(cols))
            
        histogram = np.zeros(HISTOGRAM_BINS, dtype=np.uint32)
        buf_h = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=histogram)
        
        self._kernels['histogram'](
            self.queue, g_size, l_size,
            gray_buf, buf_h, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, histogram, buf_h).wait()
        
        lut = compute_equalize_lut(histogram)
        buf_l = self._read_buffer(lut)
        
        output = np.zeros((rows, cols), dtype=np.uint8)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self._kernels['map_lut'](
            self.queue, g_size, l_size,
            gray_buf, buf_l, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

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
                ms = (time.perf_counter() - start) * 1000.0
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
        if operation == OP_GRAYSCALE:
            return self._batch_grayscale(images)
        if operation == OP_EDGES:
            return self._batch_edges(images)
        if operation == OP_BLUR:
            return self._batch_blur(images)
        if operation == OP_EQUALIZE:
            return self._batch_equalize(images)
        raise ValueError(f'Unknown operation: {operation!r}')

    def _make_batch_buffers(
        self,
        stacked_in: np.ndarray,
        out_nbytes: int,
    ) -> tuple:
        mf = cl.mem_flags
        buf_in = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=stacked_in)
        buf_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, out_nbytes)
        return buf_in, buf_out

    def _batch_global_size(
        self, shape: tuple[int, int], n: int
    ) -> tuple[int, int, int]:
        rows, cols = shape
        return (n, rows, cols)

    def _batch_grayscale(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        h, w = images[0].shape[:2]
        stacked = np.ascontiguousarray(np.stack(images))
        host_out = np.zeros((n, h, w), dtype=np.uint8)
        buf_in, buf_out = self._make_batch_buffers(stacked, host_out.nbytes)
        g_size, l_size = self._get_batch_work_sizes((h, w), n)
        self._kernels['grayscale_batch'](
            self.queue, g_size, l_size,
            buf_in, buf_out, np.int32(h), np.int32(w))
        cl.enqueue_copy(self.queue, host_out, buf_out).wait()
        return [host_out[i] for i in range(n)]

    def _batch_edges(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        h, w = images[0].shape[:2]
        
        stacked = np.ascontiguousarray(np.stack(images))
        host_gray = np.zeros((n, h, w), dtype=np.uint8)
        buf_in, buf_gray = self._make_batch_buffers(stacked, host_gray.nbytes)
        g_size, l_size = self._get_batch_work_sizes((h, w), n)
        
        self._kernels['grayscale_batch'](
            self.queue, g_size, l_size,
            buf_in, buf_gray, np.int32(h), np.int32(w))
            
        host_out = np.zeros((n, h, w), dtype=np.uint8)
        buf_out = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, host_out.nbytes)
        
        self._kernels['edges_batch'](
            self.queue, g_size, l_size,
            buf_gray, buf_out, np.int32(h), np.int32(w))
            
        cl.enqueue_copy(self.queue, host_out, buf_out).wait()
        return [host_out[i] for i in range(n)]

    def _batch_blur(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        h, w = images[0].shape[:2]
        stacked = np.ascontiguousarray(np.stack(images))
        host_out = np.zeros_like(stacked)
        buf_in, buf_out = self._make_batch_buffers(stacked, host_out.nbytes)
        g_size, l_size = self._get_batch_work_sizes((h, w), n)
        self._kernels['blur_batch'](
            self.queue, g_size, l_size,
            buf_in, buf_out, np.int32(h), np.int32(w))
        cl.enqueue_copy(self.queue, host_out, buf_out).wait()
        return [host_out[i] for i in range(n)]

    def _batch_equalize(self, images: list[np.ndarray]) -> list[np.ndarray]:
        n = len(images)
        h, w = images[0].shape[:2]
        stacked = np.ascontiguousarray(np.stack(images))
        host_gray = np.zeros((n, h, w), dtype=np.uint8)
        buf_in, buf_gray = self._make_batch_buffers(stacked, host_gray.nbytes)
        g_size, l_size = self._get_batch_work_sizes((h, w), n)
        
        self._kernels['grayscale_batch'](
            self.queue, g_size, l_size,
            buf_in, buf_gray, np.int32(h), np.int32(w))
            
        histograms = np.zeros((n, HISTOGRAM_BINS), dtype=np.uint32)
        buf_h = cl.Buffer(
            self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=histograms)
        self._kernels['histogram_batch'](
            self.queue, g_size, l_size,
            buf_gray, buf_h, np.int32(h), np.int32(w))
        cl.enqueue_copy(self.queue, histograms, buf_h).wait()
        
        luts = np.stack([compute_equalize_lut(histograms[i]) for i in range(n)])
        buf_l = self._read_buffer(luts)
        
        host_out = np.zeros((n, h, w), dtype=np.uint8)
        buf_out = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, host_out.nbytes)
        
        self._kernels['map_lut_batch'](
            self.queue, g_size, l_size,
            buf_gray, buf_l, buf_out, np.int32(h), np.int32(w))
            
        cl.enqueue_copy(self.queue, host_out, buf_out).wait()
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
        g_size, l_size = self._get_work_sizes((rows, cols))
        self._kernels['histogram'](
            self.queue, g_size, l_size,
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
        g_size, l_size = self._get_work_sizes((rows, cols))
        self._kernels['map_lut'](
            self.queue, g_size, l_size,
            gray_buf, lut_buf, out_buf, np.int32(rows), np.int32(cols))
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output
