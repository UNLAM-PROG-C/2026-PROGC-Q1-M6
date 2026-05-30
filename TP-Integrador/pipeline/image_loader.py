"""Productor: escanea una carpeta y encola rutas de imágenes."""

from __future__ import annotations

import logging
import os

from core.queue_manager import ImageQueue

IMAGE_EXTENSIONS: tuple[str, ...] = (
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tiff',
)

_LOGGER = logging.getLogger(__name__)


def scan_folder(input_dir: str) -> list[str]:
    """Escanea una carpeta y devuelve las rutas de imágenes válidas.

    Args:
        input_dir: Ruta de la carpeta de entrada.

    Returns:
        Lista ordenada de rutas con extensión en IMAGE_EXTENSIONS.
    """
    paths: list[str] = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(os.path.join(input_dir, name))
    _LOGGER.info('Imágenes encontradas: %s', len(paths))
    return paths


def enqueue_paths(
    paths: list[str],
    queue: ImageQueue,
    num_workers: int,
) -> None:
    """Encola las rutas y un sentinela None por cada worker.

    Args:
        paths: Rutas de imágenes a procesar.
        queue: Cola de entrada del pipeline.
        num_workers: Cantidad de workers que consumirán la cola.
    """
    for path in paths:
        queue.put(path)
    for _ in range(num_workers):
        queue.put(None)
    _LOGGER.info('Rutas encoladas; sentinelas: %s', num_workers)
