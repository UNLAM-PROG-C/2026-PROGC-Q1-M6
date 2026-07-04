"""Hub de progreso thread-safe que puentea hilos worker y asyncio."""

from __future__ import annotations

import asyncio
import threading
import time

from api.schemas import ProgressUpdate

ZERO: int = 0
FULL_PERCENT: float = 100.0
NO_SPEEDUP: float = 0.0


def _get_speedup() -> float:
    """Lee speedup_factor del colector activo; NO_SPEEDUP si no hay run."""
    import api.metrics_state as metrics_state
    collector = metrics_state.get_current()
    if collector is None:
        return NO_SPEEDUP
    return float(collector.get_live_stats().get('speedup_factor', NO_SPEEDUP))


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
        self._start_time: float = 0.0
        self._final_elapsed: float = 0.0
        self._final_speed: float = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Asocia el event loop y el Event de notificación asyncio."""
        self._loop = loop
        self._event = asyncio.Event()

    def update_progress(self, current: int, total: int) -> None:
        """Actualiza el progreso desde un hilo worker (API de #21).

        No toca running: ese flag lo controla exclusivamente
        set_running(), que se llama recién cuando el pipeline (incluido
        el guardado de imágenes a disco) terminó de verdad. Si no,
        current == total dispara running=False mientras el ImageSaver
        todavía está escribiendo, y el front refresca la galería antes
        de que el manifest.json nuevo esté listo.

        Args:
            current: Imágenes procesadas hasta el momento.
            total: Total de imágenes a procesar.
        """
        with self._lock:
            self._current = current
            self._total = total
            if current >= total > 0 and self._start_time > 0:
                self._final_elapsed = time.perf_counter() - self._start_time
                self._final_speed = (
                    self._current / self._final_elapsed
                    if self._final_elapsed > 0 else 0.0)
        self._notify()

    def set_running(self, running: bool) -> None:
        """Marca si hay un procesamiento en curso."""
        with self._lock:
            self._running = running
            if running:
                self._start_time = time.perf_counter()
                self._current = ZERO
                self._total = ZERO
        self._notify()

    def snapshot(self) -> ProgressUpdate:
        """Devuelve una copia consistente del estado actual.

        Returns:
            ProgressUpdate con current, total, percent y running.
        """
        with self._lock:
            elapsed = 0.0
            speed = 0.0
            eta = 0.0
            if self._running and self._start_time > 0:
                elapsed = time.perf_counter() - self._start_time
                speed = self._current / elapsed if elapsed > 0 else 0.0
                eta = (self._total - self._current) / speed if speed > 0 else 0.0
            elif not self._running and self._current > 0 and self._current == self._total:
                elapsed = self._final_elapsed
                speed = self._final_speed
                eta = 0.0

            return ProgressUpdate(
                current=self._current,
                total=self._total,
                percent=_percent(self._current, self._total),
                running=self._running,
                speed=speed,
                elapsed=elapsed,
                eta=eta,
                speedup=_get_speedup(),
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
