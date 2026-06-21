"""Tests de scan_folder y enqueue_paths del image_loader."""

from __future__ import annotations

import pathlib

from core.queue_manager import ImageQueue
from pipeline.image_loader import enqueue_paths, scan_folder

QUEUE_SIZE: int = 16
NUM_WORKERS_ONE: int = 1

VALID_FILENAMES: list[str] = ['alpha.jpg', 'beta.png', 'gamma.bmp']
INVALID_FILENAMES: list[str] = ['doc.txt', 'sheet.pdf', 'clip.mp4']
MIXED_VALID: list[str] = ['img.jpg', 'photo.png']
MIXED_INVALID: list[str] = ['notes.txt', 'video.mp4']
MIXED_FILENAMES: list[str] = MIXED_VALID + MIXED_INVALID


def _create_files(directory: pathlib.Path, names: list[str]) -> None:
    """Crea archivos vacíos en directory con los nombres dados."""
    for name in names:
        (directory / name).touch()


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
