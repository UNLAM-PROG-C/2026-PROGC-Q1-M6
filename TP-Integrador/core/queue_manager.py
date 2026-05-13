"""Cola thread-safe para el pipeline productor-consumidor de ParallelVision."""

import queue


class ImageQueue:
    """Cola thread-safe de paths de imágenes para el pipeline.

    Actúa como buffer entre el productor (carga de imágenes) y los
    consumidores (workers CPU/GPU). Envuelve queue.Queue con una
    interfaz explícita para facilitar el testing y el mocking.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """Inicializa la cola interna.

        Args:
            maxsize: Capacidad máxima. 0 significa ilimitada.
        """
        self._queue: queue.Queue[str] = queue.Queue(maxsize=maxsize)

    def put(self, item: str) -> None:
        """Encola un path de imagen. Bloquea si la cola está llena.

        Args:
            item: Path absoluto a la imagen a procesar.

        Raises:
            NotImplementedError: Implementado en feature/image-loader-queue.
        """
        raise NotImplementedError

    def get(self) -> str:
        """Desencola y retorna el siguiente path. Bloquea si está vacía.

        Returns:
            Path absoluto a la siguiente imagen a procesar.

        Raises:
            NotImplementedError: Implementado en feature/image-loader-queue.
        """
        raise NotImplementedError

    def task_done(self) -> None:
        """Señala que el item obtenido con get() fue procesado.

        Raises:
            NotImplementedError: Implementado en feature/image-loader-queue.
        """
        raise NotImplementedError

    def empty(self) -> bool:
        """Indica si la cola está vacía en este instante.

        Returns:
            True si no hay items en la cola, False en caso contrario.

        Raises:
            NotImplementedError: Implementado en feature/image-loader-queue.
        """
        raise NotImplementedError

    def qsize(self) -> int:
        """Retorna el número aproximado de items en la cola.

        Returns:
            Cantidad de items pendientes (aproximada; no usar para sync).

        Raises:
            NotImplementedError: Implementado en feature/image-loader-queue.
        """
        raise NotImplementedError
