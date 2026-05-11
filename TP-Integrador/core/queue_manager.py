"""Cola thread-safe para el pipeline productor-consumidor."""

import queue


DEFAULT_MAXSIZE: int = 0


class ImageQueue:
    """Cola sincronizada para transferir paths de imagen entre hilos.

    El modulo de carga actua como productor (put); los workers
    CPU/GPU actuan como consumidores (get / task_done).
    """

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        """Inicializa la cola interna.

        Args:
            maxsize: Limite de elementos. 0 significa ilimitado.
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: object) -> None:
        """Agrega un item a la cola (bloqueante si esta llena).

        Args:
            item: Path de imagen u objeto de trabajo.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError

    def get(self) -> object:
        """Extrae y retorna el proximo item (bloqueante si vacia).

        Returns:
            El proximo item de la cola.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError

    def task_done(self) -> None:
        """Notifica que el item obtenido con get() fue procesado.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError

    def empty(self) -> bool:
        """Retorna True si la cola no tiene items pendientes.

        Returns:
            True si la cola esta vacia, False en caso contrario.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError

    def qsize(self) -> int:
        """Retorna el numero aproximado de items en la cola.

        Returns:
            Cantidad aproximada de items pendientes.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError
