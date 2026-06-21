"""Productor: escanea una carpeta y encola rutas de imágenes."""

from __future__ import annotations

import logging
import os
import threading

from core.queue_manager import ImageQueue

IMAGE_EXTENSIONS: tuple[str, ...] = (
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tiff',
)

PRODUCER_THREAD_NAME: str = 'image-producer'

_LOGGER = logging.getLogger(__name__)


def scan_folder(
    input_dir: str,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> list[str]:
    """Escanea una carpeta y devuelve las rutas de imágenes válidas.

    Args:
        input_dir: Ruta de la carpeta de entrada.
        extensions: Extensiones de archivo a considerar.

    Returns:
        Lista ordenada de rutas con extensión en extensions.
    """
    paths: list[str] = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(extensions):
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


def load_images(
    folder: str,
    queue: ImageQueue,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> None:
    """Escanea folder y encola los paths de imágenes con un sentinel final.

    Args:
        folder: Ruta de la carpeta con imágenes.
        queue: Cola donde se encolan los paths encontrados.
        extensions: Extensiones de archivo a considerar.
    """
    paths = scan_folder(folder, extensions)
    for path in paths:
        queue.put(path)
    queue.put(None)
    _LOGGER.info('load_images: %d imágenes + sentinel encolados', len(paths))


def start_producer(
    folder: str,
    queue: ImageQueue,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> threading.Thread:
    """Lanza un hilo productor que encola las imágenes de folder.

    Args:
        folder: Ruta de la carpeta con imágenes.
        queue: Cola donde el hilo encola los paths.
        extensions: Extensiones de archivo a considerar.

    Returns:
        Hilo productor ya iniciado (daemon=True).
    """
    t = threading.Thread(
        target=load_images,
        args=(folder, queue, extensions),
        name=PRODUCER_THREAD_NAME,
        daemon=True,
    )
    t.start()
    return t
