"""WebSocket de progreso en tiempo real (#21)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.progress_hub import hub

ws_router = APIRouter()
KEEPALIVE_SECONDS: float = 15.0


async def _send_snapshot(websocket: WebSocket) -> None:
    """Envía el estado actual del hub por el WebSocket."""
    await websocket.send_text(hub.snapshot().model_dump_json())


async def _next_update(websocket: WebSocket) -> None:
    """Espera un cambio o un keepalive y reenvía el snapshot."""
    try:
        await asyncio.wait_for(hub.wait(), timeout=KEEPALIVE_SECONDS)
    except asyncio.TimeoutError:
        pass
    await _send_snapshot(websocket)


@ws_router.websocket('/ws/progress')
async def progress_ws(websocket: WebSocket) -> None:
    """Emite ProgressUpdate al cliente ante cada cambio de estado.

    Args:
        websocket: Conexión entrante del navegador.
    """
    await websocket.accept()
    await _send_snapshot(websocket)
    try:
        while True:
            await _next_update(websocket)
    except WebSocketDisconnect:
        logging.info('Cliente de progreso desconectado')
