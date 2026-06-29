"""Punto de entrada del pipeline concurrente de imágenes.

Arma el pipeline productor-consumidor, lo ejecuta e imprime un
resumen de métricas por operación.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from core.backend import VALID_OPERATIONS, CPUBackend, get_backend
from core.backend.base import CPU_BACKEND_NAME, GPUBackend
from core.metrics import CPU_BACKEND, GPU_BACKEND, MetricsCollector
from core.queue_manager import DEFAULT_QUEUE_SIZE, ImageQueue
from pipeline.image_loader import enqueue_paths, scan_folder
from pipeline.image_saver import ImageSaver
from pipeline.report_exporter import export_csv
from pipeline.result_aggregator import ResultAggregator
from pipeline.worker import MAX_WORKER_THREADS, ProcessingWorker

_LOG_FORMAT: str = '[%(levelname)s] %(message)s'
_SUMMARY_HEADER: str = '\nResumen por operación:'
IO_QUEUE_SIZE: int = 10
BENCHMARK_RECORDS_PER_IMAGE: int = 2


def _parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con input_dir, operation, workers y queue_size.
    """
    parser = argparse.ArgumentParser(description='Pipeline de imágenes')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument(
        '--operation', required=True, choices=VALID_OPERATIONS)
    parser.add_argument(
        '--workers', type=int, default=MAX_WORKER_THREADS)
    parser.add_argument(
        '--queue-size', type=int, default=DEFAULT_QUEUE_SIZE)
    parser.add_argument('--report', default=None)
    parser.add_argument('--output-dir', default='result')
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def _setup_io(
    args: argparse.Namespace,
) -> tuple[ImageQueue | None, ImageSaver | None]:
    """Crea y arranca el guardado de imágenes si no está desactivado.

    Args:
        args: Argumentos de configuración del pipeline.

    Returns:
        Tupla (io_queue, saver); ambos None si no_save está activo.
    """
    if args.no_save:
        return None, None
    os.makedirs(args.output_dir, exist_ok=True)
    io_queue: ImageQueue = ImageQueue(IO_QUEUE_SIZE)
    saver = ImageSaver(io_queue, args.output_dir)
    saver.start()
    return io_queue, saver


def _teardown_io(
    io_queue: ImageQueue | None,
    saver: ImageSaver | None,
) -> None:
    """Drena la cola de E/S y espera al ImageSaver.

    Args:
        io_queue: Cola de imágenes a guardar; None si no_save activo.
        saver: Hilo ImageSaver a detener; None si no_save activo.
    """
    if io_queue is not None:
        io_queue.put(None)
    if saver is not None:
        saver.join()


def _make_benchmark_backends(
    benchmark: bool,
) -> tuple[GPUBackend, GPUBackend | None, str | None]:
    """Devuelve (backend_primario, bench_backend, bench_label).

    En modo benchmark, el primario es CPUBackend y el bench es el
    backend GPU detectado. Si no hay GPU real, cae a modo simple.

    Args:
        benchmark: True para activar el modo benchmark CPU+GPU.

    Returns:
        Tupla (primario, bench, bench_label); bench es None sin GPU.
    """
    if not benchmark:
        return get_backend(), None, None
    gpu = get_backend()
    if gpu.backend_name == CPU_BACKEND_NAME:
        return gpu, None, None
    return CPUBackend(), gpu, GPU_BACKEND


def _warmup_backend(backend: GPUBackend | None, operation: str) -> None:
    """Llama a warmup() en el backend si está disponible."""
    if backend is None:
        return
    warmup_fn = getattr(backend, 'warmup', None)
    if warmup_fn is not None:
        warmup_fn(operation)


def _process_images(
    args: argparse.Namespace,
    backend: GPUBackend,
    input_queue: ImageQueue,
    result_queue: ImageQueue,
    paths: list[str],
    io_queue: ImageQueue | None = None,
    bench_backend: GPUBackend | None = None,
    bench_label: str | None = None,
) -> None:
    """Lanza los workers en un pool y encola las rutas a procesar.

    Args:
        args: Argumentos de configuración del pipeline.
        backend: Backend principal de procesamiento.
        input_queue: Cola de rutas de entrada.
        result_queue: Cola de resultados de salida.
        paths: Rutas de imágenes a procesar.
        io_queue: Cola de guardado de imágenes (opcional).
        bench_backend: Backend secundario para benchmark (opcional).
        bench_label: Etiqueta del bench_backend (opcional).
    """
    _warmup_backend(backend, args.operation)
    _warmup_backend(bench_backend, args.operation)
    executor = ThreadPoolExecutor(max_workers=args.workers)
    for _ in range(args.workers):
        worker = ProcessingWorker(
            input_queue, result_queue, backend, args.operation,
            backend_label=CPU_BACKEND, io_queue=io_queue,
            bench_backend=bench_backend, bench_label=bench_label)
        executor.submit(worker.run)
    enqueue_paths(paths, input_queue, args.workers)
    executor.shutdown(wait=True)


def _run_pipeline(
    args: argparse.Namespace,
    *,
    on_record: Callable[[int, int], None] | None = None,
    metrics: MetricsCollector | None = None,
    benchmark: bool = False,
) -> tuple[MetricsCollector, float, GPUBackend]:
    """Ejecuta el pipeline completo y devuelve sus métricas.

    Args:
        args: Argumentos de configuración del pipeline.
        on_record: Callback opcional invocado tras cada imagen procesada.
        metrics: Colector externo a reutilizar; se crea uno si es None.
        benchmark: True para procesar cada imagen en CPU y GPU.

    Returns:
        Tupla con el colector, el tiempo total en segundos y el backend.
    """
    metrics = metrics or MetricsCollector()
    backend, bench_b, bench_lbl = _make_benchmark_backends(benchmark)
    io_queue, saver = _setup_io(args)
    paths = scan_folder(args.input_dir)
    metrics.set_total_count(len(paths))
    recs_per = BENCHMARK_RECORDS_PER_IMAGE if bench_b is not None else 1
    input_q, result_q = ImageQueue(args.queue_size), ImageQueue(args.queue_size)
    aggregator = ResultAggregator(
        result_q, metrics, len(paths), on_record, recs_per)
    aggregator.start()
    start = time.perf_counter()
    _process_images(
        args, backend, input_q, result_q, paths, io_queue, bench_b, bench_lbl)
    result_q.put(None)
    aggregator.join()
    _teardown_io(io_queue, saver)
    return metrics, time.perf_counter() - start, backend


def _format_operation(operation: str, stats: dict) -> str:
    """Formatea una línea de resumen para una operación."""
    return (
        f'  {operation}  →  count={stats["count"]}  '
        f'avg={stats["avg_ms"]:.1f} ms  '
        f'min={stats["min_ms"]:.1f} ms  '
        f'max={stats["max_ms"]:.1f} ms')


def _print_summary(
    metrics: MetricsCollector,
    total_seconds: float,
    backend_name: str,
) -> None:
    """Imprime el resumen formateado del procesamiento.

    Args:
        metrics: Colector con las métricas acumuladas.
        total_seconds: Tiempo total de ejecución en segundos.
        backend_name: Nombre de la clase del backend activo.
    """
    summary = metrics.get_summary()
    total = sum(stats['count'] for stats in summary.values())
    throughput = metrics.get_throughput(total_seconds)
    print(f'[INFO] Backend activo: {backend_name}')
    print(
        f'[INFO] Procesadas {total} imágenes en '
        f'{total_seconds:.2f} s ({throughput:.2f} img/s)')
    print(_SUMMARY_HEADER)
    for operation, stats in summary.items():
        print(_format_operation(operation, stats))


def main() -> None:
    """Arma el pipeline, lo ejecuta e imprime el resumen final."""
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    args = _parse_args()
    metrics, total_seconds, backend = _run_pipeline(args)
    _print_summary(metrics, total_seconds, type(backend).__name__)
    if args.report:
        export_csv(metrics, args.report)
        print(f'[INFO] Reporte CSV escrito en {args.report}')


if __name__ == '__main__':
    main()
