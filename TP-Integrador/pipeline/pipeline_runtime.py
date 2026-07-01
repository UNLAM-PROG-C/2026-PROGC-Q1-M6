"""Utilidades de ejecucion para el pipeline principal."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.backend import CPUBackend
from core.metrics import CPU_BACKEND, MetricsCollector
from core.queue_manager import ImageQueue
from pipeline.image_saver import ImageSaver
from pipeline.result_aggregator import ResultAggregator
from pipeline.worker import ProcessingWorker

SAVER_QUEUE_SIZE: int = 10
SUMMARY_HEADER: str = '\nResumen por operacion:'


def create_io_resources(
    output_dir: str,
    save_enabled: bool,
) -> tuple[ImageQueue | None, ImageSaver | None]:
    """Crea la cola y el hilo de guardado si corresponde."""
    if not save_enabled:
        return None, None
    os.makedirs(output_dir, exist_ok=True)
    io_queue = ImageQueue(SAVER_QUEUE_SIZE)
    image_saver = ImageSaver(io_queue, output_dir)
    image_saver.start()
    return io_queue, image_saver


def start_workers(
    args: argparse.Namespace,
    *,
    backend: CPUBackend,
    input_queue: ImageQueue,
    result_queue: ImageQueue,
    paths: list[str],
    io_queue: ImageQueue | None,
) -> ThreadPoolExecutor:
    """Crea el pool de workers y encola las rutas a procesar."""
    executor = ThreadPoolExecutor(max_workers=args.workers)
    for _ in range(args.workers):
        worker = ProcessingWorker(
            input_queue,
            result_queue,
            backend,
            args.operation,
            backend_label=CPU_BACKEND,
            io_queue=io_queue,
        )
        executor.submit(worker.run)
    for path in paths:
        input_queue.put(path)
    for _ in range(args.workers):
        input_queue.put(None)
    return executor


def stop_workers(input_queue: ImageQueue, worker_count: int) -> None:
    """Envía un sentinel por worker."""
    for _ in range(worker_count):
        input_queue.put(None)


def cancel_executor(executor: ThreadPoolExecutor) -> None:
    """Cancela tareas pendientes y evita esperar al pool."""
    executor.shutdown(wait=False, cancel_futures=True)


def finish_pipeline(
    result_queue: ImageQueue,
    aggregator: ResultAggregator,
    io_queue: ImageQueue | None,
    image_saver: ImageSaver | None,
) -> None:
    """Cierra agregador e hilo de guardado en orden."""
    result_queue.put(None)
    aggregator.join()
    if io_queue is not None:
        io_queue.put(None)
    if image_saver is not None:
        image_saver.join()


def format_operation(operation: str, stats: dict[str, float | int]) -> str:
    """Formatea una linea del resumen final."""
    return (
        f'  {operation}  ->  count={stats["count"]}  '
        f'avg={stats["avg_ms"]:.1f} ms  '
        f'min={stats["min_ms"]:.1f} ms  '
        f'max={stats["max_ms"]:.1f} ms'
    )


def print_summary(
    metrics: MetricsCollector,
    total_seconds: float,
    backend_name: str,
) -> None:
    """Imprime el resumen final del procesamiento."""
    summary = metrics.get_summary()
    total = sum(stats['count'] for stats in summary.values())
    throughput = metrics.get_throughput(total_seconds)
    logging.info('Backend activo: %s', backend_name)
    logging.info(
        'Procesadas %d imagenes en %.2f s (%.2f img/s)',
        total,
        total_seconds,
        throughput,
    )
    logging.info(SUMMARY_HEADER)
    for operation, stats in summary.items():
        logging.info(format_operation(operation, stats))


def shutdown_pipeline(
    *,
    input_queue: ImageQueue,
    worker_count: int,
    executor: ThreadPoolExecutor,
    result_queue: ImageQueue,
    aggregator: ResultAggregator,
    io_queue: ImageQueue | None,
    image_saver: ImageSaver | None,
) -> None:
    """Aplica el shutdown limpio de workers, agregador e I/O."""
    stop_workers(input_queue, worker_count)
    cancel_executor(executor)
    finish_pipeline(result_queue, aggregator, io_queue, image_saver)


class Dashboard:
    """Facade minima para pedir confirmacion antes de iniciar."""

    def __init__(self, backend: CPUBackend, metrics: MetricsCollector) -> None:
        self.backend = backend
        self.metrics = metrics
        self.on_start: Callable[[argparse.Namespace], Any] | None = None

    def run(self, cfg: argparse.Namespace) -> None:
        """Muestra el backend activo y espera la confirmacion del usuario."""
        logging.info(
            'Backend activo: %s - %s',
            type(self.backend).__name__,
            self.backend.device_info,
        )
        logging.info(
            'Presione ENTER para iniciar el pipeline, o Q + ENTER para salir',
        )
        choice = input().strip().lower()
        if choice == 'q':
            logging.info('Cancelado por el usuario')
            return
        if self.on_start is not None:
            self.on_start(cfg)
