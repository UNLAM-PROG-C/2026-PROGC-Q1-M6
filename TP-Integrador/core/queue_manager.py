"""Cola thread-safe para el pipeline productor-consumidor.

Envuelve ``queue.Queue`` con una interfaz acotada y explícita, sin
exponer la cola interna al resto del pipeline.
"""

from __future__ import annotations

import queue
from typing import Any

DEFAULT_QUEUE_SIZE: int = 32


class ImageQueue:
    """Cola thread-safe acotada para paths de imágenes y resultados."""

    def __init__(self, maxsize: int) -> None:
        """Inicializa la cola interna con un tamaño máximo.

        Args:
            maxsize: Capacidad máxima de la cola.
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any) -> None:
        """Encola un elemento, bloqueando si la cola está llena."""
        self._queue.put(item)

    def get(self) -> Any:
        """Desencola un elemento, bloqueando si la cola está vacía."""
        return self._queue.get()

    def task_done(self) -> None:
        """Señala que un elemento obtenido con get() fue procesado."""
        self._queue.task_done()

    def join(self) -> None:
        """Bloquea hasta que todos los elementos sean procesados."""
        self._queue.join()

    def empty(self) -> bool:
        """Indica si la cola está vacía en este instante."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Devuelve el tamaño aproximado de la cola."""
        return self._queue.qsize()
