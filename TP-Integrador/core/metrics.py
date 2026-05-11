"""Agregador de resultados con acceso thread-safe."""

import threading


class MetricsCollector:
    """Recolecta y resume metricas de procesamiento de imagenes.

    Usa threading.Lock para garantizar escritura segura cuando
    multiples workers reportan resultados en paralelo.
    """

    def __init__(self) -> None:
        """Inicializa el lock y la estructura de datos interna."""
        self._lock: threading.Lock = threading.Lock()
        self._data: list = []

    def record(
        self,
        image_name: str,
        backend: str,
        operation: str,
        elapsed_ms: float,
    ) -> None:
        """Registra el resultado de procesar una imagen.

        Args:
            image_name: Nombre del archivo de imagen procesado.
            backend: Identificador del backend usado ('cpu', 'cuda', 'opencl').
            operation: Transformacion aplicada (ej: 'grayscale').
            elapsed_ms: Tiempo de procesamiento en milisegundos.

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError

    def get_summary(self) -> dict:
        """Retorna un resumen agregado de todas las metricas registradas.

        Returns:
            Diccionario con estadisticas de procesamiento por backend
            y operacion (conteo, tiempo promedio, speedup, etc.).

        Raises:
            NotImplementedError: Hasta que se complete la implementacion.
        """
        raise NotImplementedError
