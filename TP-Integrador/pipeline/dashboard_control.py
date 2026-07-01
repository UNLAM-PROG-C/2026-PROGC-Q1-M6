"""Control de procesos del dashboard opcional."""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

FRONTEND_URL: str = 'http://localhost:5173'
FRONTEND_DIR: Path = Path(__file__).resolve().parents[1] / 'frontend'
UVICORN_MODULE: str = 'api.main:app'
FRONTEND_PORT: int = 5173
FRONTEND_TIMEOUT_SECONDS: int = 60
POLL_INTERVAL_SECONDS: float = 0.5


def start_uvicorn() -> subprocess.Popen | None:
    """Inicia uvicorn como subproceso."""
    try:
        cmd = [sys.executable, '-m', 'uvicorn', UVICORN_MODULE, '--reload']
        return subprocess.Popen(cmd)
    except OSError:
        logging.exception('No se pudo iniciar uvicorn')
        return None


def wait_for_port(url: str, timeout: int) -> None:
    """Abre el navegador cuando el frontend responde."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            with socket.create_connection(
                ('localhost', FRONTEND_PORT),
                timeout=1,
            ):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(POLL_INTERVAL_SECONDS)


# pylint: disable=consider-using-with
def start_frontend() -> subprocess.Popen | None:
    """Inicia el frontend de desarrollo."""
    npm_path = shutil.which('npm')
    if npm_path is None:
        logging.warning('npm no encontrado en PATH; frontend no iniciado')
        return None
    try:
        cmd = [npm_path, 'run', 'dev']
        # Popen debe quedarse vivo para mantener el frontend en desarrollo.
        process = subprocess.Popen(cmd, cwd=FRONTEND_DIR)
        logging.info('Frontend dev iniciado (pid=%s)', process.pid)
        wait_for_port(FRONTEND_URL, FRONTEND_TIMEOUT_SECONDS)
        return process
    except FileNotFoundError:
        logging.error('npm no encontrado en el sistema: %s', npm_path)
    except OSError:
        logging.exception('No se pudo iniciar el dev server del frontend')
    logging.info(
        'Inicia manualmente el frontend: cd %s && npm install && npm run dev',
        FRONTEND_DIR,
    )
    return None


def start_dashboard_processes() -> list[subprocess.Popen]:
    """Arranca backend API y frontend, si es posible."""
    processes: list[subprocess.Popen] = []
    uvicorn_process = start_uvicorn()
    if uvicorn_process is not None:
        processes.append(uvicorn_process)
    frontend_process = start_frontend()
    if frontend_process is not None:
        processes.append(frontend_process)
    return processes


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    """Termina los procesos de dashboard que fueron iniciados."""
    for process in processes:
        try:
            logging.info('Terminando proceso pid=%s', process.pid)
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
