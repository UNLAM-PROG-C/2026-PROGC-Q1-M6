"""Agregador de métricas de procesamiento con acceso thread-safe."""

import threading


class MetricsCollector:
    """Recolecta métricas de rendimiento de forma thread-safe.

    Usa threading.Lock para proteger la lista interna de registros
    contra condiciones de carrera cuando múltiples workers escriben
    simultáneamente.
    """

    def __init__(self) -> None:
        """Inicializa el lock y la lista de registros."""
        self._lock: threading.Lock = threading.Lock()
        self._data: list[dict[str, object]] = []

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
            backend: Identificador del backend ('cpu', 'cuda', 'opencl').
            operation: Transformación aplicada (VALID_OPERATIONS).
            elapsed_ms: Tiempo de procesamiento en milisegundos.

        Raises:
            NotImplementedError: Implementado en feature/metrics-report.
        """
        raise NotImplementedError

    def get_summary(self) -> dict[str, object]:
        """Retorna un resumen agregado de todas las métricas registradas.

        Returns:
            Diccionario con totales y promedios por backend y operación.

        Raises:
            NotImplementedError: Implementado en feature/metrics-report.
        """
        raise NotImplementedError
