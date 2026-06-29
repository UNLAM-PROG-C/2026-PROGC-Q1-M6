"""Estado compartido del MetricsCollector del run activo."""

from __future__ import annotations

import threading

from core.metrics import MetricsCollector

_lock: threading.Lock = threading.Lock()
_current: MetricsCollector | None = None
_output_dir: str = ''


def set_current(metrics: MetricsCollector, output_dir: str) -> None:
    """Registra el colector activo del run actual.

    Args:
        metrics: Colector recién creado para el run.
        output_dir: Directorio de salida del run.
    """
    global _current, _output_dir
    with _lock:
        _current = metrics
        _output_dir = output_dir


def get_current() -> MetricsCollector | None:
    """Devuelve el colector del run activo; None si no hay run."""
    with _lock:
        return _current


def get_output_dir() -> str:
    """Devuelve el directorio de salida del run activo."""
    with _lock:
        return _output_dir


def clear() -> None:
    """Limpia el estado tras el run."""
    global _current, _output_dir
    with _lock:
        _current = None
        _output_dir = ''
