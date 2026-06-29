"""Modelos Pydantic compartidos por la API y el frontend."""

from __future__ import annotations

from pydantic import BaseModel, Field

MIN_WORKERS: int = 1
MAX_WORKERS: int = 16


class BackendInfo(BaseModel):
    """Información del backend de procesamiento detectado."""

    backend_name: str
    device_info: str
    is_gpu: bool


class Operation(BaseModel):
    """Operación de imagen con su etiqueta en español."""

    value: str
    label: str


class DirEntry(BaseModel):
    """Subdirectorio listado por el explorador de carpetas."""

    name: str
    path: str
    is_dir: bool = True


class BrowseResult(BaseModel):
    """Resultado de navegar una carpeta del filesystem del servidor."""

    path: str
    parent: str | None
    entries: list[DirEntry]


class StartConfig(BaseModel):
    """Configuración enviada al iniciar un procesamiento."""

    input_dir: str
    output_dir: str
    operation: str = Field(...)
    workers: int = Field(ge=MIN_WORKERS, le=MAX_WORKERS)


class StartResponse(BaseModel):
    """Acuse de recibo del inicio de procesamiento (stub Fase 2)."""

    accepted: bool
    message: str


class ProgressUpdate(BaseModel):
    """Estado de progreso emitido por el WebSocket."""

    current: int
    total: int
    percent: float
    running: bool
    speed: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0
    speedup: float = 0.0


class ChartPoint(BaseModel):
    """Punto de la serie temporal CPU vs GPU por lote."""

    batch: int
    cpu_ms: float
    gpu_ms: float


class OperationResult(BaseModel):
    """Resultado por operación con tiempos y speedup."""

    operation: str
    cpu_avg_ms: float
    gpu_avg_ms: float
    speedup: float
