"""Exportación de métricas de sesión a CSV."""

from __future__ import annotations

import csv
import os

from core.metrics import MetricsCollector

CSV_HEADER: tuple[str, ...] = (
    'image_name', 'backend', 'operation', 'elapsed_ms')


def export_csv(metrics: MetricsCollector, output_path: str) -> None:
    """Exporta los registros de la sesión a un archivo CSV.

    Args:
        metrics: Colector con los registros crudos a exportar.
        output_path: Ruta del archivo CSV de destino.
    """
    _ensure_parent_dir(output_path)
    with open(output_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(metrics.get_records())


def _ensure_parent_dir(output_path: str) -> None:
    """Crea la carpeta de destino si no existe."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
