"""
Escribir un test de integración que verifique el pipeline CPU completo
de extremo a extremo, sin mocks de los componentes principales.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from core.backend import CPUBackend
from core.metrics import CPU_BACKEND
from core.metrics import MetricsCollector
from core.queue_manager import ImageQueue
from pipeline.image_loader import enqueue_paths, scan_folder
from pipeline.result_aggregator import ResultAggregator
from pipeline.worker import ProcessingWorker

MAX_WORKERS = 5
MAX_IMAGENES_TEST_FULL_CPU_PIPELINE = 5
MAX_IMAGENES_TEST_INVALID_IMAGES = 3
MAX_IMAGE_SIZE = 10

# Inputs
OPERATION_NAME_TEST = "grayscale"


def crear_imagenes_con_numpy(carpeta, cantidad):
    """Crea imágenes PNG de prueba en la carpeta especificada."""
    for i in range(cantidad):
        img_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img_path = carpeta + f"\\test_image_{i}.png"
        Image.fromarray(img_array).save(img_path)


def _process_images(
    backend: CPUBackend,
    input_queue: ImageQueue,
    result_queue: ImageQueue,
    paths: list[str],
) -> None:
    """Lanza los workers en un pool y encola las rutas a procesar."""
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    for _ in range(MAX_WORKERS):
        worker = ProcessingWorker(
            input_queue, result_queue, backend, OPERATION_NAME_TEST,
            backend_label=CPU_BACKEND)
        executor.submit(worker.run)
    enqueue_paths(paths, input_queue, MAX_WORKERS)
    executor.shutdown(wait=True)


def _run_pipeline(test_dir: str):
    """Ejecuta el pipeline completo sobre un directorio y devuelve métricas."""
    image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    result_image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    backend = CPUBackend()
    metrics_collector = MetricsCollector()
    result_aggregator = ResultAggregator(result_image_queue, metrics_collector)

    paths = scan_folder(test_dir)
    result_aggregator.start()
    _process_images(backend, image_queue, result_image_queue, paths)
    result_image_queue.put(None)
    result_aggregator.join()

    return metrics_collector.get_records()


def test_full_cpu_pipeline(tmp_path):
    """Prueba de integración de pipeline CPU completo."""
    test_dir = str(tmp_path)
    crear_imagenes_con_numpy(test_dir, MAX_IMAGENES_TEST_FULL_CPU_PIPELINE)

    start = time.perf_counter()
    metrics = _run_pipeline(test_dir)

    assert len(metrics) == MAX_IMAGENES_TEST_FULL_CPU_PIPELINE, \
        "Faltan resultados, deben ser 1 por cada imagen."
    for metric in metrics:
        assert metric.elapsed_ms > 0, \
            "Todos los tiempos registrados deben ser mayores a 0 ms."
    assert (time.perf_counter() - start) < 30, \
        "El pipeline debería completarse en menos de 30 segundos."


def test_pipeline_with_empty_folder(tmp_path):
    """Verifica que el pipeline termine correctamente sin imágenes."""
    metrics = _run_pipeline(str(tmp_path))
    assert len(metrics) == 0, \
        "No se esperaban resultados ya que la carpeta estaba vacía."


def test_pipeline_invalid_images_skipped(tmp_path):
    """Verifica que el pipeline ignore archivos no válidos como .txt."""
    test_dir = str(tmp_path)
    crear_imagenes_con_numpy(test_dir, MAX_IMAGENES_TEST_INVALID_IMAGES)

    for i in range(2):
        with open(
            f"{test_dir}\\invalid_file_{i}.txt", 'w', encoding='utf-8'
        ) as f:
            f.write("")

    metrics = _run_pipeline(test_dir)

    assert len(metrics) == MAX_IMAGENES_TEST_INVALID_IMAGES, \
        "Se esperaban resultados solo para las imágenes válidas."
