"""Aplicación FastAPI: monta routers REST y el WebSocket de progreso."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from api.progress_hub import hub
from api.progress_ws import ws_router
from api.routes import init_backend, router

_LOG_FORMAT: str = '[%(levelname)s] %(message)s'

app = FastAPI(title='ParallelVision Dashboard API')
app.include_router(router)
app.include_router(ws_router)


@app.on_event('startup')
async def _on_startup() -> None:
    """Detecta el backend y enlaza el hub al event loop activo."""
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    hub.bind_loop(asyncio.get_running_loop())
    info = init_backend()
    logging.info('Backend activo: %s (%s)',
                 info.backend_name, info.device_info)
