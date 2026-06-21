"""Tests de image_loader: scan_folder, enqueue_paths, load_images y start_producer."""

from __future__ import annotations

import pathlib
import threading

from core.queue_manager import ImageQueue
from pipeline.image_loader import (
    PRODUCER_THREAD_NAME,
    enqueue_paths,
    load_images,
    scan_folder,
    start_producer,
)

QUEUE_SIZE: int = 16
NUM_WORKERS_ONE: int = 1

VALID_FILENAMES: list[str] = ['alpha.jpg', 'beta.png', 'gamma.bmp']
INVALID_FILENAMES: list[str] = ['doc.txt', 'sheet.pdf', 'clip.mp4']
MIXED_VALID: list[str] = ['img.jpg', 'photo.png']
MIXED_INVALID: list[str] = ['notes.txt', 'video.mp4']
MIXED_FILENAMES: list[str] = MIXED_VALID + MIXED_INVALID

_QUEUE_MAXSIZE: int = 50
_IMAGE_COUNT: int = 3


def _create_files(directory: pathlib.Path, names: list[str]) -> None:
    """Crea archivos vacíos en directory con los nombres dados."""
    for name in names:
        (directory / name).touch()


def _make_images(tmp_path: pathlib.Path, count: int = _IMAGE_COUNT) -> list[str]:
    """Crea archivos .jpg falsos en tmp_path y retorna sus rutas ordenadas."""
    paths = []
    for i in range(count):
        p = tmp_path / f'img_{i:02d}.jpg'
        p.write_bytes(b'fake')
        paths.append(str(p))
    return sorted(paths)


def test_scan_empty_folder(tmp_path: pathlib.Path) -> None:
    """Carpeta vacía devuelve lista vacía."""
    result = scan_folder(str(tmp_path))
    assert not result


def test_scan_valid_extensions(tmp_path: pathlib.Path) -> None:
    """jpg, png y bmp son reconocidos como imágenes válidas."""
    _create_files(tmp_path, VALID_FILENAMES)
    result = scan_folder(str(tmp_path))
    assert len(result) == len(VALID_FILENAMES)


def test_scan_ignore_invalid_extensions(tmp_path: pathlib.Path) -> None:
    """txt, pdf y mp4 no son incluidos en el resultado."""
    _create_files(tmp_path, INVALID_FILENAMES)
    result = scan_folder(str(tmp_path))
    assert not result


def test_scan_mixed_extensions(tmp_path: pathlib.Path) -> None:
    """Mezcla de extensiones: solo las válidas aparecen en el resultado."""
    _create_files(tmp_path, MIXED_FILENAMES)
    result = scan_folder(str(tmp_path))
    assert len(result) == len(MIXED_VALID)


def test_sentinel_always_enqueued(tmp_path: pathlib.Path) -> None:
    """None es siempre el último ítem encolado por enqueue_paths."""
    _create_files(tmp_path, VALID_FILENAMES)
    paths = scan_folder(str(tmp_path))
    img_queue: ImageQueue = ImageQueue(maxsize=QUEUE_SIZE)
    enqueue_paths(paths, img_queue, NUM_WORKERS_ONE)
    total = len(paths) + NUM_WORKERS_ONE
    items: list[str | None] = [img_queue.get() for _ in range(total)]
    assert items[-1] is None


def test_start_producer_returns_started_thread(tmp_path: pathlib.Path) -> None:
    q = ImageQueue(maxsize=_QUEUE_MAXSIZE)
    t = start_producer(str(tmp_path), q)
    assert isinstance(t, threading.Thread)
    assert t.ident is not None
    t.join(timeout=5)


def test_producer_is_daemon(tmp_path: pathlib.Path) -> None:
    q = ImageQueue(maxsize=_QUEUE_MAXSIZE)
    t = start_producer(str(tmp_path), q)
    assert t.daemon is True
    t.join(timeout=5)


def test_producer_thread_name(tmp_path: pathlib.Path) -> None:
    q = ImageQueue(maxsize=_QUEUE_MAXSIZE)
    t = start_producer(str(tmp_path), q)
    assert t.name == PRODUCER_THREAD_NAME
    t.join(timeout=5)


def test_load_images_enqueues_paths_and_sentinel(tmp_path: pathlib.Path) -> None:
    expected = _make_images(tmp_path)
    q = ImageQueue(maxsize=_QUEUE_MAXSIZE)
    t = threading.Thread(target=load_images, args=(str(tmp_path), q))
    t.start()
    t.join(timeout=5)
    received = [q.get() for _ in range(len(expected) + 1)]
    assert received[:-1] == expected
    assert received[-1] is None


def test_start_producer_ends_cleanly(tmp_path: pathlib.Path) -> None:
    _make_images(tmp_path)
    q = ImageQueue(maxsize=_QUEUE_MAXSIZE)
    t = start_producer(str(tmp_path), q)
    t.join(timeout=5)
    assert not t.is_alive()
