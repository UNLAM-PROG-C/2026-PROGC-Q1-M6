# Decisión de arquitectura: Dashboard React + FastAPI (en vez de Tkinter)

> Requerido por el issue **#20**: documentar la decisión si la GUI no usa Tkinter.

## Contexto

La consigna original planteaba la **Capa 5 (Dashboard en tiempo real)** con
**Tkinter/PyQt + matplotlib**. Para esta entrega se decidió implementarla como
una **SPA desacoplada** — **React + Vite + TypeScript + TailwindCSS + shadcn/ui**
en `frontend/` — que consume una **API FastAPI** (`api/`), con el progreso en
vivo viajando por **WebSocket**.

## Motivación

1. **Separación de capas real.** El backend Python (concurrencia: workers,
   backends, métricas) y el frontend (presentación) quedan totalmente
   independientes y comunicados por un contrato HTTP/WebSocket explícito. Eso
   refuerza el objetivo pedagógico de la materia: la **concurrencia** vive en
   Python y no se mezcla con el loop de eventos de una GUI de escritorio.
2. **El progreso en tiempo real encaja con el modelo de hilos.** Los hilos
   worker sólo escriben estado bajo `Lock` en `ProgressHub` y notifican al event
   loop con `loop.call_soon_threadsafe(...)`; el WebSocket empuja los cambios al
   navegador. Es el equivalente moderno al *polling* de la GUI, sin condiciones
   de carrera y sin que el worker toque la red.
3. **Portabilidad.** Una SPA corre en cualquier navegador, sin depender de Tk en
   el sistema operativo de cada integrante.
4. **Calidad de UI** con un sistema de componentes accesibles (Radix/shadcn) y
   theming consistente, difícil de igualar con Tkinter.

## Trade-off asumido

Es la opción de **mayor alcance**: agrega un toolchain Node, una capa API y un
canal WebSocket. A cambio, el **núcleo del TP no se toca**: `api/` es una capa
delgada de **lectura** sobre `core/` y `pipeline/`. En esta fase la ejecución
real del pipeline desde el botón "Iniciar" y los datos en vivo quedan para
**Fase 3 (#28)**; acá se construye el esqueleto y el canal de progreso con un
emisor *stub*.

## Arquitectura resultante

```
core/      → contratos compartidos (sin tocar): get_backend, VALID_OPERATIONS…
pipeline/  → workers, cola, agregador (sin tocar)
api/       → FastAPI: /api/backend, /api/operations, /api/browse, /api/start,
             WS /ws/progress  (capa de exposición, Python)
frontend/  → React + Vite + TS + Tailwind + shadcn (presentación)
```

### Contratos reutilizados (sin modificar)

- `get_backend()` → backend activo con `backend_name` y `device_info`.
- `CPU_BACKEND_NAME` → distingue GPU (badge verde) de CPU (badge ámbar).
- `VALID_OPERATIONS` → operaciones válidas; la API les mapea etiquetas en español.
- `MetricsCollector` → contrato thread-safe (lo usará la Fase 3).

## Cómo correrlo

```bash
# 1) Backend (desde TP-Integrador/)
pip install -r requirements.txt
uvicorn api.main:app --reload          # http://localhost:8000

# 2) Frontend (en otra terminal)
cd frontend
npm install
npm run dev                            # http://localhost:5173 (proxy → 8000)
```
