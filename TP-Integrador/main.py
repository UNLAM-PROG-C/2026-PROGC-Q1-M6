"""Punto de entrada del pipeline concurrente de imágenes.

Arma el pipeline productor-consumidor, lo ejecuta e imprime un
resumen de métricas por operación.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from core.backend import VALID_OPERATIONS, CPUBackend, get_backend
from core.metrics import CPU_BACKEND, MetricsCollector
from core.queue_manager import DEFAULT_QUEUE_SIZE, ImageQueue
from pipeline.image_loader import enqueue_paths, scan_folder
from pipeline.report_exporter import export_csv
from pipeline.result_aggregator import ResultAggregator
from pipeline.worker import MAX_WORKER_THREADS, ProcessingWorker

_LOG_FORMAT: str = '[%(levelname)s] %(message)s'
_SUMMARY_HEADER: str = '\nResumen por operación:'


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
    return parser.parse_args()


def _process_images(
    args: argparse.Namespace,
    backend: CPUBackend,
    input_queue: ImageQueue,
    result_queue: ImageQueue,
    paths: list[str],
) -> None:
    """Lanza los workers en un pool y encola las rutas a procesar."""
    executor = ThreadPoolExecutor(max_workers=args.workers)
    for _ in range(args.workers):
        worker = ProcessingWorker(
            input_queue, result_queue, backend, args.operation,
            backend_label=CPU_BACKEND)
        executor.submit(worker.run)
    enqueue_paths(paths, input_queue, args.workers)
    executor.shutdown(wait=True)


def _run_pipeline(
    args: argparse.Namespace,
) -> tuple[MetricsCollector, float, CPUBackend]:
    """Ejecuta el pipeline completo y devuelve sus métricas.

    Args:
        args: Argumentos de configuración del pipeline.

    Returns:
        Tupla con el colector, el tiempo total en segundos y el
        backend usado.
    """
    backend = get_backend()
    metrics = MetricsCollector()
    input_queue = ImageQueue(args.queue_size)
    result_queue = ImageQueue(args.queue_size)
    paths = scan_folder(args.input_dir)
    aggregator = ResultAggregator(result_queue, metrics)
    aggregator.start()
    start = time.perf_counter()
    _process_images(args, backend, input_queue, result_queue, paths)
    result_queue.put(None)
    aggregator.join()
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
