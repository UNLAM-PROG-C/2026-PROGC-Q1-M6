"""Hub de progreso thread-safe que puentea hilos worker y asyncio."""

from __future__ import annotations

import asyncio
import threading

from api.schemas import ProgressUpdate

ZERO: int = 0
FULL_PERCENT: float = 100.0


def _percent(current: int, total: int) -> float:
    """Calcula el porcentaje de avance evitando división por cero."""
    if total <= ZERO:
        return 0.0
    return min(current / total * FULL_PERCENT, FULL_PERCENT)


class ProgressHub:
    """Estado de progreso compartido entre workers y el WebSocket.

    Los hilos worker sólo escriben números bajo un ``Lock`` y avisan al
    event loop con ``call_soon_threadsafe``; nunca tocan la red.
    """

    def __init__(self) -> None:
        """Inicializa el estado protegido por Lock."""
        self._lock: threading.Lock = threading.Lock()
        self._current: int = ZERO
        self._total: int = ZERO
        self._running: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Asocia el event loop y el Event de notificación asyncio."""
        self._loop = loop
        self._event = asyncio.Event()

    def update_progress(self, current: int, total: int) -> None:
        """Actualiza el progreso desde un hilo worker (API de #21).

        Args:
            current: Imágenes procesadas hasta el momento.
            total: Total de imágenes a procesar.
        """
        with self._lock:
            self._current = current
            self._total = total
            self._running = current < total
        self._notify()

    def set_running(self, running: bool) -> None:
        """Marca si hay un procesamiento en curso."""
        with self._lock:
            self._running = running
        self._notify()

    def snapshot(self) -> ProgressUpdate:
        """Devuelve una copia consistente del estado actual.

        Returns:
            ProgressUpdate con current, total, percent y running.
        """
        with self._lock:
            return ProgressUpdate(
                current=self._current,
                total=self._total,
                percent=_percent(self._current, self._total),
                running=self._running,
            )

    async def wait(self) -> None:
        """Espera a que haya un nuevo estado de progreso."""
        if self._event is None:
            return
        await self._event.wait()
        self._event.clear()

    def _notify(self) -> None:
        """Despierta al event loop de forma thread-safe."""
        if self._loop is None or self._event is None:
            return
        self._loop.call_soon_threadsafe(self._event.set)


hub = ProgressHub()
