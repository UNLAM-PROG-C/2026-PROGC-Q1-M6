"""Endpoints REST de la API (backend, operaciones, browse, start)."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

from fastapi import APIRouter, Query

from core.backend import CPU_BACKEND_NAME, VALID_OPERATIONS, get_backend
from core.backend.base import GPUBackend
from api.schemas import (
    BackendInfo,
    BrowseResult,
    DirEntry,
    Operation,
    StartConfig,
    StartResponse,
)

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
    """Resuelve ``raw`` confinándolo a BROWSE_ROOT."""
    if not raw:
        return BROWSE_ROOT
    target = Path(raw).resolve()
    if target == BROWSE_ROOT or BROWSE_ROOT in target.parents:
        return target
    return BROWSE_ROOT


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
    """Navega los subdirectorios del filesystem del servidor.

    Args:
        path: Ruta absoluta a inspeccionar; vacía usa la raíz.

    Returns:
        BrowseResult con la ruta actual, su padre y subdirectorios.
    """
    target = _safe_resolve(path)
    parent = None if target == BROWSE_ROOT else str(target.parent)
    return BrowseResult(
        path=str(target), parent=parent, entries=_list_subdirs(target))


@router.post('/start', status_code=202)
def start(config: StartConfig) -> StartResponse:
    """Valida la configuración y acusa recibo (stub de Fase 2).

    Args:
        config: Carpetas, operaciones e hilos elegidos en la SPA.

    Returns:
        StartResponse confirmando la recepción de la configuración.
    """
    logging.info(
        'Solicitud de inicio: %s -> %s, ops=%s, workers=%d',
        config.input_dir, config.output_dir,
        config.operations, config.workers)
    return StartResponse(
        accepted=True,
        message='Configuración recibida (ejecución real en Fase 3).')
