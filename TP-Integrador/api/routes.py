"""Endpoints REST de la API (backend, operaciones, browse, start)."""

from __future__ import annotations

import functools
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse

import api.metrics_state as metrics_state
from api.progress_hub import hub
from api.schemas import (
    BackendInfo,
    BrowseResult,
    DirEntry,
    Operation,
    StartConfig,
    StartResponse,
)
from core.backend import CPU_BACKEND_NAME, VALID_OPERATIONS, get_backend
from core.backend.base import GPUBackend
from core.metrics import MetricsCollector
from pipeline.image_loader import scan_folder

_RUN_LOCK: threading.Lock = threading.Lock()
_RUN_STATE: dict[str, bool] = {'running': False}


def _acquire_run() -> bool:
    """Marca pipeline en ejecución; devuelve False si ya está corriendo."""
    with _RUN_LOCK:
        if _RUN_STATE['running']:
            return False
        _RUN_STATE['running'] = True
        return True


def _release_run() -> None:
    """Libera el guard de concurrencia del pipeline."""
    with _RUN_LOCK:
        _RUN_STATE['running'] = False

OPERATION_LABELS: dict[str, str] = {
    'grayscale': 'Escala de grises',
    'edges': 'Detección de bordes',
    'blur': 'Desenfoque',
    'equalize': 'Ecualización',
}
BROWSE_ROOT: Path = Path(__file__).resolve().parent.parent

router = APIRouter(prefix='/api')


@functools.lru_cache(maxsize=1)
def _cached_backend() -> GPUBackend:
    """Detecta el backend una sola vez y lo memoiza."""
    return get_backend()


def init_backend() -> BackendInfo:
    """Devuelve la info del backend activo (detección cacheada).

    Returns:
        BackendInfo con el backend activo del servidor.
    """
    return _to_backend_info(_cached_backend())


def _to_backend_info(backend: GPUBackend) -> BackendInfo:
    """Convierte un GPUBackend en su DTO serializable."""
    return BackendInfo(
        backend_name=backend.backend_name,
        device_info=backend.device_info,
        is_gpu=backend.backend_name != CPU_BACKEND_NAME,
    )


@router.get('/backend')
def read_backend() -> BackendInfo:
    """Devuelve la información del backend activo (cacheado)."""
    return init_backend()


@router.get('/operations')
def read_operations() -> list[Operation]:
    """Devuelve las operaciones válidas con etiquetas en español."""
    return [
        Operation(value=value, label=OPERATION_LABELS[value])
        for value in VALID_OPERATIONS
    ]


def _safe_resolve(raw: str) -> Path:
    """Resuelve ``raw`` permitiendo cualquier ruta."""
    if not raw:
        return Path.cwd().resolve()
    return Path(raw).resolve()


def _list_subdirs(target: Path) -> list[DirEntry]:
    """Lista los subdirectorios de ``target`` ordenados por nombre."""
    try:
        children = sorted(target.iterdir(), key=lambda item: item.name)
    except PermissionError:
        return []
    return [
        DirEntry(name=child.name, path=str(child))
        for child in children
        if child.is_dir()
    ]


@router.get('/browse')
def browse(path: str = Query(default='')) -> BrowseResult:
    """Navega los subdirectorios del filesystem del servidor."""
    target = _safe_resolve(path)
    parent = None if target.parent == target else str(target.parent)
    entries = _list_subdirs(target)
    
    import os
    if os.name == 'nt' and target.parent == target:
        import string
        drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        for drive in drives:
            if drive.rstrip('\\') != str(target).rstrip('\\'):
                entries.append(DirEntry(name=f"Drive {drive}", path=drive))

    return BrowseResult(
        path=str(target), parent=parent, entries=entries)


_QUEUE_SIZE: int = 10


def _build_pipeline_args(config: StartConfig) -> SimpleNamespace:
    """Construye el Namespace de argumentos del pipeline desde la config.

    Args:
        config: Configuración enviada por la SPA al iniciar.

    Returns:
        SimpleNamespace compatible con la firma de _run_pipeline.
    """
    return SimpleNamespace(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        operation=config.operation,
        workers=config.workers,
        queue_size=_QUEUE_SIZE,
        no_save=False,
    )


def _validate_start(config: StartConfig) -> None:
    """Valida la configuración antes de iniciar el pipeline.

    Raises:
        HTTPException: 400 si el directorio no existe, la operación es
            inválida, o no hay imágenes en el directorio.
    """
    if not Path(config.input_dir).is_dir():
        raise HTTPException(status_code=400, detail='Directorio no encontrado.')
    if config.operation not in VALID_OPERATIONS:
        raise HTTPException(status_code=400, detail='Operación inválida.')
    if not scan_folder(config.input_dir):
        raise HTTPException(status_code=400, detail='Sin imágenes en el directorio.')


def _run_pipeline_task(config: StartConfig) -> None:
    """Ejecuta el pipeline en un hilo de fondo."""
    from main import _run_pipeline
    mc = MetricsCollector()
    metrics_state.set_current(mc, config.output_dir)
    args = _build_pipeline_args(config)
    hub.set_running(True)
    try:
        _, secs, _ = _run_pipeline(
            args, on_record=hub.update_progress,
            metrics=mc, benchmark=True)
        logging.info('Pipeline finalizado en %.2fs', secs)
    except Exception as e:
        logging.error('Error en pipeline: %s', e)
    finally:
        hub.set_running(False)
        _release_run()


@router.post('/start', status_code=202)
def start(config: StartConfig, background_tasks: BackgroundTasks) -> StartResponse:
    """Inicia el procesamiento delegando a un background task.

    Args:
        config: Configuración elegida en la SPA.
        background_tasks: Inyector de tareas en segundo plano de FastAPI.

    Returns:
        StartResponse confirmando la recepción.
    """
    logging.info(
        'Solicitud de inicio: %s -> %s, op=%s, workers=%d',
        config.input_dir, config.output_dir,
        config.operation, config.workers)

    _validate_start(config)
    if not _acquire_run():
        raise HTTPException(status_code=409, detail='Pipeline ya en ejecución.')

    background_tasks.add_task(_run_pipeline_task, config)
    
    return StartResponse(
        accepted=True,
        message='Procesamiento iniciado.')


@router.get('/results')
def get_results(output_dir: str = Query(...)) -> dict:
    """Lee el manifest.json de la carpeta de salida y devuelve los resultados."""
    target_dir = _safe_resolve(output_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=400, detail="Directorio de salida inválido.")
    
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        return {"results": []}
        
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Error leyendo manifest.json: %s", e)
        raise HTTPException(status_code=500, detail="Error leyendo el manifiesto.")


@router.get('/image')
def get_image(dir_path: str = Query(...), filename: str = Query(...)) -> FileResponse:
    """Sirve una imagen estática de forma segura."""
    target_dir = _safe_resolve(dir_path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=400, detail="Directorio inválido.")
        
    file_path = target_dir / filename
    
    # Prevenir Path Traversal validando que resolve() siga dentro de target_dir
    resolved_file = file_path.resolve()
    if target_dir not in resolved_file.parents:
        raise HTTPException(status_code=403, detail="Acceso denegado.")
        
    if not resolved_file.exists() or not resolved_file.is_file():
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
        
    return FileResponse(resolved_file)
