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
import subprocess
import sys
import shutil
import socket
import webbrowser

from core.backend import VALID_OPERATIONS, CPUBackend, get_backend
from core.metrics import CPU_BACKEND, MetricsCollector
from core.queue_manager import DEFAULT_QUEUE_SIZE, ImageQueue
from pipeline.image_loader import enqueue_paths, scan_folder
from pipeline.image_saver import ImageSaver
from pipeline.report_exporter import export_csv
from pipeline.result_aggregator import ResultAggregator
from pipeline.worker import MAX_WORKER_THREADS, ProcessingWorker
from typing import Any

_LOG_FORMAT: str = '[%(levelname)s] %(message)s'
_SUMMARY_HEADER: str = '\nResumen por operación:'
FRONTEND_DEV_URL: str = 'http://localhost:5173'
FRONTEND_DIR: str = 'frontend'
UVICORN_MODULE: str = 'api.main:app'
FRONTEND_PORT: int = 5173


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
    parser.add_argument('--dashboard', action='store_true',
                        help='Inicia la API y el frontend del dashboard')
    return parser.parse_args()


def _start_uvicorn() -> subprocess.Popen | None:
    """Inicia uvicorn como subproceso y devuelve el proceso o None."""
    try:
        cmd = [sys.executable, '-m', 'uvicorn', UVICORN_MODULE, '--reload']
        p = subprocess.Popen(cmd)
        logging.info('uvicorn iniciado (pid=%s)', p.pid)
        return p
    except OSError:
        logging.exception('No se pudo iniciar uvicorn')
        return None


def _open_url_when_ready(url: str, timeout: int = 60) -> None:
    """Abre `url` en el navegador cuando el puerto responda (timeout segundos)."""
    host = 'localhost'
    port = FRONTEND_PORT
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=1):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)


def _start_frontend() -> subprocess.Popen | None:
    """Inicia `npm run dev` en el directorio del frontend y abre el URL."""
    npm_path = shutil.which('npm')
    if npm_path is None:
        logging.warning('npm no encontrado en PATH; frontend no iniciado')
        return None
    try:
        cmd = [npm_path, 'run', 'dev']
        p = subprocess.Popen(cmd, cwd=os.path.join(os.getcwd(), FRONTEND_DIR))
        logging.info('Frontend dev iniciado (pid=%s)', p.pid)
        _open_url_when_ready(FRONTEND_DEV_URL)
        return p
    except FileNotFoundError:
        logging.error('npm no encontrado en el sistema: %s', npm_path)
        logging.info('Inicia manualmente el frontend: cd %s && npm install && npm run dev',
                     FRONTEND_DIR)
        return None
    except OSError:
        logging.exception('No se pudo iniciar el dev server del frontend')
        logging.info('Inicia manualmente el frontend: cd %s && npm install && npm run dev',
                     FRONTEND_DIR)
        return None


def _start_dashboard_processes() -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    p = _start_uvicorn()
    if p is not None:
        procs.append(p)
    f = _start_frontend()
    if f is not None:
        procs.append(f)
    return procs


def _process_images(
    args: argparse.Namespace,
    backend: CPUBackend,
    input_queue: ImageQueue,
    result_queue: ImageQueue,
    paths: list[str],
    io_queue: ImageQueue | None = None,
) -> None:
    """Lanza los workers en un pool y encola las rutas a procesar."""
    executor = ThreadPoolExecutor(max_workers=args.workers)
    for _ in range(args.workers):
        worker = ProcessingWorker(
            input_queue, result_queue, backend, args.operation,
            backend_label=CPU_BACKEND, io_queue=io_queue)
        executor.submit(worker.run)
    enqueue_paths(paths, input_queue, args.workers)
    # Wrap executor in a simple manager for clear shutdown API
    return WorkerManager(executor, args.workers)


class WorkerManager:
    """Encapsula el ThreadPoolExecutor y ofrece `shutdown()`."""

    def __init__(self, executor: ThreadPoolExecutor, worker_count: int) -> None:
        self._executor = executor
        self._worker_count = worker_count

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

def _shutdown_pipeline(
    worker_manager: WorkerManager | None,
    aggregator: ResultAggregator | None,
    input_queue: ImageQueue | None,
    io_queue: ImageQueue | None = None,
) -> None:
    """Shutdown en orden inverso: señal, esperar workers, esperar agregador."""
    if input_queue is not None and worker_manager is not None:
        # send one sentinel per worker
        for _ in range(worker_manager._worker_count):
            input_queue.put(None)
    if worker_manager is not None:
        worker_manager.shutdown(wait=True)
    if aggregator is not None:
        aggregator.join()
    if io_queue is not None:
        io_queue.put(None)


class Dashboard:
    """Minimal dashboard facade: simplifies main for interactive runs.

    This is a CLI placeholder that blocks on `run()` and calls `on_start`
    when the user requests to start processing. Designed to be small and
    replaceable by a real GUI later.
    """

    def __init__(self, backend: CPUBackend, metrics: MetricsCollector) -> None:
        self.backend = backend
        self.metrics = metrics
        self.on_start: Callable[[argparse.Namespace], Any] | None = None

    def run(self, cfg: argparse.Namespace) -> None:
        logging.info('Backend activo: %s - %s', type(self.backend).__name__,
                     self.backend.device_info)
        logging.info('Presione ENTER para iniciar el pipeline, o Q + ENTER para salir')
        choice = input().strip().lower()
        if choice == 'q':
            logging.info('Cancelado por el usuario')
            return
        if self.on_start is not None:
            self.on_start(cfg)


def _run_pipeline(
    args: argparse.Namespace,
    *,
    on_record: Callable[[int, int], None] | None = None,
) -> tuple[MetricsCollector, float, CPUBackend]:
    """Ejecuta el pipeline completo y devuelve sus métricas.

    Args:
        args: Argumentos de configuración del pipeline.
        on_record: Callback opcional invocado tras cada imagen procesada.

    Returns:
        Tupla con el colector, el tiempo total en segundos y el
        backend usado.
    """
    backend = get_backend()
    metrics = MetricsCollector()
    input_queue = ImageQueue(args.queue_size)
    result_queue = ImageQueue(args.queue_size)
    
    io_queue = None
    image_saver = None
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        io_queue = ImageQueue(10)
        image_saver = ImageSaver(io_queue, args.output_dir)
        image_saver.start()

    paths = scan_folder(args.input_dir)
    aggregator = ResultAggregator(result_queue, metrics, len(paths), on_record)
    aggregator.start()
    start = time.perf_counter()
    executor = _process_images(args, backend, input_queue, result_queue, paths, io_queue)
    # wait for workers, support KeyboardInterrupt for clean shutdown
    try:
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        logging.info('Interrupción recibida: iniciando shutdown limpio')
        # notify workers to stop by sending sentinels
        for _ in range(args.workers):
            input_queue.put(None)
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)
        logging.info('Estado al cierre: %s', metrics.get_live_stats())
        raise
    finally:
        result_queue.put(None)
        aggregator.join()
    
    if io_queue is not None:
        io_queue.put(None)
    if image_saver is not None:
        image_saver.join()
        
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
    logging.info('[INFO] Backend activo: %s', backend_name)
    logging.info('[INFO] Procesadas %d imágenes en %.2f s (%.2f img/s)',
                    total, total_seconds, throughput)
    logging.info(_SUMMARY_HEADER)
    for operation, stats in summary.items():
        logging.info(_format_operation(operation, stats))


def main() -> None:
    """Arma el pipeline, lo ejecuta e imprime el resumen final."""
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    args = _parse_args()

    backend = get_backend()
    metrics = MetricsCollector()

    # Optional progress hub (used by API/websocket)
    hub = None
    try:
        from api.progress_hub import hub as progress_hub

        hub = progress_hub
    except ImportError:
        hub = None

    if hub is not None:
        hub.set_running(False)

    procs: list[subprocess.Popen] = []

    # Build a minimal Dashboard facade and attach the start handler
    dashboard = Dashboard(backend, metrics)

    def _start_handler(cfg: argparse.Namespace) -> None:
        # Create queues and optional IO saver
        input_queue = ImageQueue(cfg.queue_size)
        result_queue = ImageQueue(cfg.queue_size)
        io_queue = None
        image_saver = None
        if not cfg.no_save:
            os.makedirs(cfg.output_dir, exist_ok=True)
            io_queue = ImageQueue(10)
            image_saver = ImageSaver(io_queue, cfg.output_dir)
            image_saver.start()

        paths = scan_folder(cfg.input_dir)
        metrics.set_total_count(len(paths))
        aggregator = ResultAggregator(result_queue, metrics, len(paths), on_record=(hub.update_progress if hub is not None else None))
        aggregator.start()

        # start workers
        worker_manager = _process_images(cfg, backend, input_queue, result_queue, paths, io_queue)

        try:
            # wait for workers to finish
            worker_manager.shutdown(wait=True)
        except KeyboardInterrupt:
            logging.info('Interrupción recibida: iniciando shutdown limpio')
            _shutdown_pipeline(worker_manager, aggregator, input_queue, io_queue)
            raise
        finally:
            # ensure aggregator and IO shutdown
            _shutdown_pipeline(worker_manager, aggregator, input_queue, io_queue)

        # print summary and export if requested
        total_seconds = metrics.get_live_stats().get('elapsed_seconds', 0.0)
        _print_summary(metrics, float(total_seconds), type(backend).__name__)
        if cfg.report:
            export_csv(metrics, cfg.report)
            logging.info('Reporte CSV escrito en %s', cfg.report)

    dashboard.on_start = _start_handler

    try:
        if args.dashboard:
            procs = _start_dashboard_processes()

        # Run the dashboard (blocks until user closes the UI or finishes)
        dashboard.run(args)
    except KeyboardInterrupt:
        logging.info('Interrupción por usuario: detenido')
    finally:
        if hub is not None:
            hub.set_running(False)
        # Terminate any dashboard subprocesses we started
        for p in procs:
            try:
                logging.info('Terminando proceso pid=%s', p.pid)
                p.terminate()
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except OSError:
                    pass
            except OSError:
                try:
                    p.kill()
                except OSError:
                    pass


if __name__ == '__main__':
    main()
