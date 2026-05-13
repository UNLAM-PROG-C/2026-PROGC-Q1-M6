# CLAUDE.md — ParallelVision

Guía de contexto para el asistente de IA trabajando en este proyecto.

## Proyecto

**ParallelVision** — pipeline de procesamiento masivo de imágenes con CPU y GPU.  
Materia: Programación Concurrente, UNLAM, 1° Cuatrimestre 2026.  
Entrega: 01/07–08/07/2026.

## Arquitectura

El sistema tiene 5 capas:

```
Capa 1: Carga de imágenes      → escanea carpeta, encola paths
Capa 2: Cola de trabajo (Queue) → thread-safe, productor-consumidor
Capa 3: Workers CPU + GPU       → ThreadPoolExecutor | CUDA | OpenCL
Capa 4: Agregador de resultados → threading.Lock, métricas, speedup
Capa 5: Dashboard en tiempo real → Tkinter/PyQt + matplotlib
```

### Patrón Strategy para backend GPU

Al iniciar, la app detecta hardware en orden CUDA → OpenCL → CPU. Todos los
backends exponen la misma interfaz `GPUBackend.process(image, operation)`.

```
parallelvision/
├── core/
│   ├── backend.py          # GPUBackend base + get_backend() + CUDABackend/OpenCLBackend/CPUBackend
│   ├── queue_manager.py    # ImageQueue (productor-consumidor)
│   └── metrics.py          # MetricsCollector (thread-safe con Lock)
├── pipeline/
│   └── worker.py           # ThreadPoolExecutor workers
├── gui/
│   └── dashboard.py        # UI Tkinter/PyQt + gráfico matplotlib
└── main.py
```

### Módulos por integrante

| Rama | Integrante | Módulo |
|---|---|---|
| `feature/image-loader-queue` | **Tomás Felice** | Carga de imágenes, Cola, Interfaz gráfica |
| `feature/cpu-pipeline-backend` | Compañero 1 | Pipeline CPU, detección de backend, `GPUBackend` base |
| `feature/gpu-cuda-opencl` | Compañero 2 | Backend CUDA (Numba) + OpenCL (PyOpenCL), sync CPU-GPU |
| `feature/metrics-report` | Compañero 3 | MetricsCollector, reporte CSV, gráfico speedup |

Los archivos `core/backend.py`, `core/queue_manager.py` y `core/metrics.py` son
**contratos de interfaz compartidos** — no modificar sin coordinar con el equipo.

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

- `main` siempre contiene código funcional. Push directo prohibido.
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
| Lenguaje | Python 3.11+ |
| Concurrencia CPU | `concurrent.futures.ThreadPoolExecutor` |
| Procesamiento imagen | Pillow, OpenCV |
| GPU NVIDIA | Numba (CUDA kernels) |
| GPU AMD/Intel | PyOpenCL |
| Cola | `queue.Queue` |
| Sync | `threading.Lock`, `threading.Semaphore` |
| GUI | Tkinter o PyQt5 |
| Gráficos | matplotlib |
| Linting | pylint |
