"""Implementación del backend de procesamiento para OpenCL."""

from __future__ import annotations

import numpy as np

from core.backend.base import (
    GPUBackend,
    OPENCL_BACKEND_NAME,
    _gpu_semaphore,
    MAX_PIXEL_VALUE,
)
from core.backend.cpu import _to_grayscale

try:
    import pyopencl as cl
    _OPENCL_AVAILABLE = True
except ImportError:
    cl = None
    _OPENCL_AVAILABLE = False


OPENCL_GRAYSCALE_RED_WEIGHT: float = 0.299
OPENCL_GRAYSCALE_GREEN_WEIGHT: float = 0.587
OPENCL_GRAYSCALE_BLUE_WEIGHT: float = 0.114
OPENCL_IMAGE_CHANNELS: int = 3
OPENCL_EDGE_MARGIN: int = 1
OPENCL_SOBEL_WEIGHT: float = 2.0

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
    """
    # pylint: enable=line-too-long
    
    _OPERATIONS_OPENCL = {
        'grayscale': 'grayscale',
        'edges': 'edges',
    }
else:
    _OPENCL_KERNELS_SOURCE = ""
    _OPERATIONS_OPENCL = {}


class OpenCLBackend(GPUBackend):
    """Procesa imágenes en GPU AMD/Intel mediante kernels OpenCL."""

    def __init__(self):
        if not _OPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL no está disponible")
            
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No se encontraron plataformas OpenCL")
            
        device = platforms[0].get_devices()[0]
        self.backend_name = OPENCL_BACKEND_NAME
        self.device_info = device.name
        
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, _OPENCL_KERNELS_SOURCE).build()

    def _blur(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Operación 'blur' no implementada en OpenCL")

    def _equalize(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Operación 'equalize' no implementada en OpenCL")

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        if operation == 'blur':
            return self._blur(image)
        if operation == 'equalize':
            return self._equalize(image)
            
        if operation not in _OPERATIONS_OPENCL:
            raise ValueError(f'Unknown operation: {operation!r}')
            
        with _gpu_semaphore:
            return self._run_kernel(image, operation)

    def _create_buffers(self, image: np.ndarray, output: np.ndarray) -> tuple:
        """Crea los buffers de lectura y escritura para OpenCL."""
        mf = cl.mem_flags
        image_contig = np.ascontiguousarray(image)
        img_buf = cl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_contig
        )
        out_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes)
        return img_buf, out_buf

    def _run_kernel(self, image: np.ndarray, operation: str) -> np.ndarray:
        if operation == "edges":
            image = _to_grayscale(image)
            
        rows, cols = image.shape[:2]
        output = np.zeros((rows, cols), dtype=np.uint8)
        
        img_buf, out_buf = self._create_buffers(image, output)
        
        kernel_name = _OPERATIONS_OPENCL[operation]
        kernel_func = getattr(self.program, kernel_name)
        
        kernel_func(
            self.queue, (rows, cols), None, 
            img_buf, out_buf, np.int32(rows), np.int32(cols)
        )
            
        cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output

