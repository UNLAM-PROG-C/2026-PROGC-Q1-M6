"""Punto de entrada del pipeline concurrente de imagenes.

Arma el pipeline productor-consumidor, lo ejecuta e imprime un
resumen de metricas por operacion.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from core.backend import CPUBackend, VALID_OPERATIONS, get_backend
from core.metrics import CPU_BACKEND, MetricsCollector
from core.queue_manager import DEFAULT_QUEUE_SIZE, ImageQueue
from pipeline.dashboard_control import (
    start_dashboard_processes,
    terminate_processes,
)
from pipeline.image_loader import scan_folder
from pipeline.report_exporter import export_csv
from pipeline.result_aggregator import ResultAggregator
from pipeline.pipeline_runtime import (
    Dashboard,
    create_io_resources,
    finish_pipeline,
    print_summary,
    shutdown_pipeline,
    start_workers,
)
from pipeline.worker import MAX_WORKER_THREADS

try:
    from api.progress_hub import hub as progress_hub
except ImportError:
    progress_hub = None

_LOG_FORMAT: str = '[%(levelname)s] %(message)s'


def _parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description='Pipeline de imagenes')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--operation', required=True, choices=VALID_OPERATIONS)
    parser.add_argument('--workers', type=int, default=MAX_WORKER_THREADS)
    parser.add_argument('--queue-size', type=int, default=DEFAULT_QUEUE_SIZE)
    parser.add_argument('--report', default=None)
    parser.add_argument('--output-dir', default='result')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Inicia la API y el frontend del dashboard',
    )
    return parser.parse_args()


def _run_pipeline_session(
    args: argparse.Namespace,
    backend: CPUBackend,
    metrics: MetricsCollector,
    hub: Any | None,
) -> None:
    """Ejecuta el pipeline completo y asegura el shutdown limpio."""
    input_queue = ImageQueue(args.queue_size)
    result_queue = ImageQueue(args.queue_size)
    io_queue, image_saver = create_io_resources(
        args.output_dir,
        not args.no_save,
    )
    paths = scan_folder(args.input_dir)
    metrics.set_total_count(len(paths))
    aggregator = ResultAggregator(
        result_queue,
        metrics,
        len(paths),
        None if hub is None else hub.update_progress,
    )
    aggregator.start()
    start_time = time.perf_counter()
    executor = start_workers(
        args,
        backend=backend,
        input_queue=input_queue,
        result_queue=result_queue,
        paths=paths,
        io_queue=io_queue,
    )
    interrupted = False
    try:
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        interrupted = True
        logging.info('Interrupcion recibida: iniciando shutdown limpio')
        shutdown_pipeline(
            input_queue=input_queue,
            worker_count=args.workers,
            executor=executor,
            result_queue=result_queue,
            aggregator=aggregator,
            io_queue=io_queue,
            image_saver=image_saver,
        )
        logging.info('Estado al cierre: %s', metrics.get_live_stats())
        raise
    finally:
        if not interrupted:
            finish_pipeline(result_queue, aggregator, io_queue, image_saver)
    print_summary(
        metrics,
        time.perf_counter() - start_time,
        type(backend).__name__,
    )
    if args.report:
        export_csv(metrics, args.report)
        logging.info('Reporte CSV escrito en %s', args.report)


def run_pipeline(args: argparse.Namespace) -> None:
    """Delega la orquestacion completa."""
    # Optional progress hub (used by API/websocket)
    hub = progress_hub
    if hub is not None:
        hub.set_running(False)

    backend = get_backend()
    metrics = MetricsCollector()
    dashboard = Dashboard(backend, metrics)
    dashboard.on_start = lambda cfg: _run_pipeline_session(
        cfg,
        backend,
        metrics,
        hub,
    )

    procs: list[Any] = []
    try:
        if args.dashboard:
            procs = start_dashboard_processes()
        dashboard.run(args)
    except KeyboardInterrupt:
        logging.info('Interrupcion por usuario: detenido')
    finally:
        if hub is not None:
            hub.set_running(False)
        terminate_processes(procs)


def main() -> None:
    """Punto de entrada principal del sistema."""
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    # Build a minimal Facade and execute the pipeline.
    run_pipeline(_parse_args())


if __name__ == '__main__':
    main()
