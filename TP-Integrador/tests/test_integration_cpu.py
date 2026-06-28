#Escribir un test de integración que verifique el pipeline CPU completo de extremo a extremo, sin mocks de los componentes principales.

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from pipeline.worker import ProcessingWorker
from core.backend import CPUBackend
from pipeline.result_aggregator import ResultAggregator
from pipeline.image_loader import enqueue_paths, scan_folder
from core.metrics import MetricsCollector
from core.queue_manager import ImageQueue
from core.metrics import CPU_BACKEND
import numpy as np
import os

MAX_WORKERS = 5
MAX_IMAGENES_TEST_FULL_CPU_PIPELINE = 5
MAX_IMAGENES_TEST_INVALID_IMAGES = 3
MAX_IMAGE_SIZE = 10

#Inputs
PATH_FULL_CPU_PIPELINE_FOLDER = r"public\\images\\test_full_cpu_pipeline"
PATH_EMPTY_FOLDER = r"public\\images\\test_empty_folder"
PATH_INVALID_IMAGES_FOLDER = r"public\\images\\test_imagenes_invalidas"
OPERATION_NAME_TEST = "grayscale"

def crear_imagenes_con_numpy(carpeta, cantidad):
    """Crea imágenes PNG de prueba en la carpeta especificada."""
    from PIL import Image
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

def test_full_cpu_pipeline():
    # Crear carpeta temporal para las imágenes
    test_dir = PATH_FULL_CPU_PIPELINE_FOLDER
    os.mkdir(test_dir) if not os.path.exists(test_dir) else None

    # Generar 5 imágenes PNG de prueba (50x50 px)
    crear_imagenes_con_numpy(test_dir, MAX_IMAGENES_TEST_FULL_CPU_PIPELINE)

    # Instanciar componentes del pipeline
    image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    result_image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    backend = CPUBackend()
    paths = scan_folder(test_dir)
    metrics_collector = MetricsCollector()
    result_aggregator = ResultAggregator( result_image_queue, metrics_collector)

    result_aggregator.start()
    start = time.perf_counter()
    _process_images(backend, image_queue, result_image_queue, paths)    
    # Señalar que no habrá más imágenes
    result_image_queue.put(None)  # Usar None como señal de finalización

    # Esperar a que el pipeline termine
    result_aggregator.join() #worker.join(timeout=30)  # Esperar máximo 30 segundos

    # Verificar resultados en MetricsCollector
    metrics = metrics_collector.get_records()

    assert len(metrics) == MAX_IMAGENES_TEST_FULL_CPU_PIPELINE , "Se esperaban resultados para todas las operaciones por cada imagen."

    for metric in metrics:
        assert metric.elapsed_ms > 0, "Todos los tiempos registrados deben ser mayores a 0 ms."
    
    assert (time.perf_counter() - start) < 30, "El pipeline debería completarse en menos de 30 segundos."


def test_pipeline_with_empty_folder():
    # Crear carpeta vacía para la prueba
    test_dir = PATH_EMPTY_FOLDER
    os.mkdir(test_dir) if not os.path.exists(test_dir) else None
    paths = scan_folder(test_dir)

    # Instanciar componentes del pipeline
    image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    result_image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    backend = CPUBackend()
    metrics_collector = MetricsCollector()

    result_aggregator = ResultAggregator( result_image_queue, metrics_collector)

    result_aggregator.start()
    _process_images(backend, image_queue, result_image_queue, paths)
    result_image_queue.put(None)  # Señal de finalización

    # Esperar a que el pipeline termine
    result_aggregator.join()

    # Verificar que no se procesaron imágenes
    metrics = metrics_collector.get_records()
    assert len(metrics) == 0, "No se esperaban resultados ya que la carpeta estaba vacía."


def test_pipeline_invalid_images_skipped():
    # Crear carpeta con imágenes válidas y archivos inválidos
    test_dir = PATH_INVALID_IMAGES_FOLDER
    os.mkdir(test_dir) if not os.path.exists(test_dir) else None
    # Generar 3 imágenes PNG válidas
    crear_imagenes_con_numpy(test_dir, MAX_IMAGENES_TEST_INVALID_IMAGES)
    # Crear 2 archivos .txt inválidos
    for i in range(2):
        with open(test_dir + "\\" + f"invalid_file_{i}.txt", 'w') as f:
            f.write("")

    paths = scan_folder(test_dir)

    # Instanciar componentes del pipeline
    image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    result_image_queue = ImageQueue(maxsize=MAX_IMAGE_SIZE)
    backend = CPUBackend()
    metrics_collector = MetricsCollector()
    result_aggregator = ResultAggregator(result_image_queue, metrics_collector)

    # Arrancar el hilo del worker
    result_aggregator.start()
    _process_images(backend, image_queue, result_image_queue, paths)
    result_image_queue.put(None)  # Señal de finalización
    result_aggregator.join()    # Esperar a que el pipeline termine

    # Verificar resultados en MetricsCollector
    metrics = metrics_collector.get_records()

    assert len(metrics) == MAX_IMAGENES_TEST_INVALID_IMAGES, "Se esperaban resultados solo para las imágenes válidas."