# CLAUDE.md — ParallelVision

Guía de contexto para el asistente de IA trabajando en este proyecto.

## Proyecto

**ParallelVision** — pipeline de procesamiento masivo de imágenes con CPU y GPU.  
Materia: Programación Concurrente, UNLAM, 1° Cuatrimestre 2026.  
Entrega: 08/07/2026.

Se usa de dos formas: **CLI headless** (`python main.py`) y **dashboard web**
cliente-servidor (backend FastAPI + frontend React/Vite, progreso por WebSocket).

## Arquitectura

El sistema tiene 5 capas:

```
Capa 1: Carga de imágenes       → escanea carpeta, encola paths (pipeline/image_loader.py)
Capa 2: Cola de entrada (Queue) → thread-safe, productor-consumidor (core/queue_manager.py)
Capa 3: Workers CPU + GPU       → ThreadPoolExecutor | CUDA | OpenCL, batching GPU por lotes
                                  + cola de E/S y hilo ImageSaver que guarda a disco
Capa 4: Agregador de resultados → threading.Lock, métricas, speedup, reporte CSV
Capa 5: Dashboard en tiempo real → API FastAPI + WebSocket (api/) + React/Vite (frontend/)
```

### Patrón Strategy para backend GPU

Al iniciar, la app detecta hardware en orden CUDA → OpenCL → CPU. Todos los
backends exponen la misma interfaz `GPUBackend.process(image, operation)` y
`process_batch(images, operation)`.

```
TP-Integrador/
├── core/
│   ├── backend/            # paquete Strategy (contrato compartido)
│   │   ├── base.py         # GPUBackend (ABC) + VALID_OPERATIONS + constantes GPU + semáforo
│   │   ├── factory.py      # get_backend() — detección CUDA → OpenCL → CPU
│   │   ├── cuda.py         # CUDABackend (Numba CUDA kernels)
│   │   ├── opencl.py       # OpenCLBackend (PyOpenCL kernels)
│   │   └── cpu.py          # CPUBackend (OpenCV/Pillow, fallback)
│   ├── queue_manager.py    # ImageQueue (productor-consumidor)
│   └── metrics.py          # MetricsCollector (thread-safe con Lock)
├── pipeline/
│   ├── image_loader.py     # escanea carpeta y encola paths (Capa 1)
│   ├── worker.py           # ProcessingWorker sobre ThreadPoolExecutor
│   ├── gpu_batcher.py      # GpuBatcher — agrupa imágenes en lotes GPU
│   ├── image_saver.py      # ImageSaver — hilo de guardado a disco (cola de E/S)
│   ├── result_aggregator.py# ResultAggregator — consume resultados y actualiza métricas
│   └── report_exporter.py  # export_csv() del reporte de speedup
├── api/                    # FastAPI: routes REST, metrics_routes, progress_ws (WebSocket)
├── frontend/               # SPA React + Vite + TypeScript + Tailwind (dashboard)
├── colab/                  # notebook para correr el backend CUDA en Colab (T4)
├── scripts/                # utilidades (ej. descargar_imagenes.py)
└── main.py                 # entrypoint CLI headless
```

## Reglas de código (no negociables)

### Límites de la cátedra

- **Máximo 15 líneas por función/método** (regla de la cátedra).
- **Sin números mágicos** — toda constante debe tener nombre en CAPS_WITH_UNDER.
- **Usar patrones de diseño** (Strategy ya definido; Producer-Consumer en queue).
- `pylint` debe ejecutarse sin warnings no justificados **antes de abrir un PR**.

### Estilo (Google Python Style Guide adaptado)

- Python 3.11+. Indentación: 4 espacios. Línea máxima: **80 caracteres**.
- Sin punto y coma al final de línea. Sin backslash para continuar línea (usar paréntesis).
- **Imports**: siempre absolutos (`from parallelvision.core import backend`), nunca relativos.
  Orden: `__future__` → stdlib → terceros → proyecto. Un grupo por línea en blanco.
- **Anotaciones de tipo** obligatorias en toda la API pública.
- **Docstrings** en formato Google (Args / Returns / Raises) para funciones públicas y
  cualquier función de más de 10 líneas.
- **Threading**: usar `Queue` para comunicación entre hilos, `Lock` para recursos
  compartidos, `Semaphore` para limitar lotes GPU. No asumir atomicidad de built-ins.
- **Logging**: siempre `logging.info('msg: %s', var)`, nunca f-string en logging.
- Mutable defaults prohibidos: usar `None` y asignar dentro de la función.
- Excepciones específicas, nunca `except:` o `except Exception: pass`.

### Nomenclatura

| Tipo | Convención |
|---|---|
| Clases | `CapWords` |
| Funciones / métodos | `lower_with_under()` |
| Constantes | `CAPS_WITH_UNDER` |
| Privados/internos | prefijo `_` |
| Parámetros / variables locales | `lower_with_under` |

### Ejemplo de función bien escrita

```python
MAX_GPU_BATCH_SIZE: int = 64
VALID_OPERATIONS = ('grayscale', 'edges', 'blur', 'equalize')


def process_image(
    image: np.ndarray,
    operation: str,
    backend: GPUBackend,
) -> np.ndarray:
    """Aplica la operación indicada a la imagen usando el backend activo.

    Args:
        image: Array NumPy con shape (H, W, C), dtype uint8.
        operation: Transformación a aplicar. Valores: VALID_OPERATIONS.
        backend: Backend de procesamiento activo (CUDA, OpenCL o CPU).

    Returns:
        Array procesado con el mismo shape que la entrada.

    Raises:
        ValueError: Si operation no es un valor válido.
    """
    if operation not in VALID_OPERATIONS:
        raise ValueError(f'Unknown operation: {operation!r}')
    return backend.process(image, operation)
```

## Convenciones de commits (Conventional Commits en español)

```
<tipo>: <descripción en imperativo, minúsculas, sin punto final>
```

Tipos: `feat` | `fix` | `refactor` | `test` | `docs` | `chore` | `perf`

```bash
feat: implementar detección automática de backend CUDA
fix: corregir condición de carrera en agregador de resultados
docs: agregar manual de usuario al README
chore: agregar numba y pyopencl a requirements.txt
```

Para cerrar un issue: incluir `Closes #N` en el cuerpo del commit o en la descripción del PR.

## Ramas (GitHub Flow)

- La rama de integración es **`develop`** (base de los PR y la entrega); siempre
  contiene código funcional. Push directo prohibido.
- Todo cambio entra por PR con al menos **1 revisión aprobada**.
- Nomenclatura: `<tipo>/<descripcion-en-kebab-case>` (minúsculas, guiones).
  - `feature/image-loader-queue`, `fix/queue-deadlock`, `docs/readme-manual`

## Pull Requests

Checklist antes de abrir PR:

- [ ] `pylint parallelvision/` sin warnings no justificados
- [ ] Ninguna función supera 15 líneas
- [ ] Sin números mágicos
- [ ] Probado localmente
- [ ] El código no rompe otras funcionalidades

Plantilla de descripción de PR:

```markdown
## ¿Qué se implementó?
## ¿Cómo probarlo?
## Issues relacionados
Closes #N
## Checklist
- [ ] pylint limpio
- [ ] funciones ≤ 15 líneas
- [ ] sin números mágicos
- [ ] probado localmente
```

## Tecnologías

| Área | Herramienta |
|---|---|
| Lenguaje backend | Python 3.11+ |
| Lenguaje frontend | TypeScript (~5.7) |
| Concurrencia CPU | `concurrent.futures.ThreadPoolExecutor` |
| Procesamiento imagen | NumPy, Pillow |
| GPU NVIDIA | Numba (CUDA kernels) |
| GPU AMD/Intel | PyOpenCL |
| Cola | `queue.Queue` |
| Sync | `threading.Lock`, `threading.Semaphore` |
| API / servidor | FastAPI + Uvicorn + WebSockets |
| GUI (dashboard) | React 18 + Vite + TailwindCSS |
| Gráficos | Recharts |
| Túnel Colab | pyngrok |
| Linting / tests | pylint, pytest |
