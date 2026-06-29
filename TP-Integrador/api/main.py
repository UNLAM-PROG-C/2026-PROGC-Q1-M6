"""Aplicación FastAPI: monta routers REST y el WebSocket de progreso."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from api.metrics_routes import metrics_router
from api.progress_hub import hub
from api.progress_ws import ws_router
from api.routes import init_backend, router

_LOG_FORMAT: str = '[%(levelname)s] %(message)s'


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Detecta el backend y enlaza el hub al event loop activo."""
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    hub.bind_loop(asyncio.get_running_loop())
    info = init_backend()
    logging.info('Backend activo: %s (%s)',
                 info.backend_name, info.device_info)
    yield


app = FastAPI(title='ParallelVision Dashboard API', lifespan=_lifespan)
app.include_router(router)
app.include_router(ws_router)
app.include_router(metrics_router)
