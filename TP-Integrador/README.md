# ParallelVision 👁️🚀

Pipeline de procesamiento masivo de imágenes con CPU + GPU

**Universidad Nacional de La Matanza (UNLaM)**
Programación Concurrente · 1° Cuatrimestre 2026 · Trabajo Práctico Integrador
Fecha de entrega: **08/07/2026**

## Integrantes

| Nombre | DNI |
| --- | --- |
| Felice, Tomás Agustín | 44.789.809 |
| De La Cruz Zamudio, Axel Nahuel | 41.063.583 |
| Graneros, Brian Ariel | 41.130.084 |

### Tabla de compatibilidad de Hardware

El sistema tiene detección automática de hardware, utilizando el mejor disponible según la siguiente prioridad:

| Hardware | Backend | Librería | Prioridad |
|---|---|---|---|
| NVIDIA GPU | CUDA | Numba | 1 (Más Alta) |
| AMD / Intel GPU | OpenCL | PyOpenCL | 2 |
| Cualquier CPU | Multithreading | NumPy / concurrent.futures | 3 (Más Baja) |

---

## 1. Descripción y finalidad

**ParallelVision** es un pipeline de procesamiento masivo de imágenes diseñado para aplicar
diversas operaciones y filtros a grandes volúmenes de archivos de forma eficiente.

El problema que resuelve es el **cuello de botella** que se genera al procesar miles de imágenes
de manera secuencial. Para solucionarlo, ParallelVision aprovecha al máximo los recursos de
hardware disponibles, aplicando técnicas de concurrencia y combinando **procesamiento asíncrono
en la CPU** con **aceleración por GPU**.

El usuario apunta el sistema a una carpeta de imágenes, elige una transformación y el sistema la
aplica en paralelo sobre todos los archivos. Las operaciones soportadas son:

- `grayscale` — escala de grises.
- `edges` — detección de bordes (Sobel).
- `blur` — desenfoque gaussiano.
- `equalize` — ecualización de histograma.

Dos características centrales lo diferencian:

- **Detección automática de backend en runtime:** al iniciar, la aplicación detecta qué hardware
  hay disponible y selecciona el mejor motor de procesamiento (CUDA → OpenCL → CPU) sin que el
  usuario configure nada. Funciona en cualquier máquina, con o sin GPU.
- **Dashboard de speedup en tiempo real:** una interfaz web muestra el progreso imagen por imagen
  y una comparativa cuantitativa de rendimiento CPU vs GPU, ilustrando de forma tangible el
  beneficio del paralelismo masivo.

---

## 2. Lenguajes y versiones

| Componente | Lenguaje | Versión |
| --- | --- | --- |
| Backend (pipeline + API) | Python | **3.11 o superior** (probado en 3.12) |
| Frontend (dashboard) | TypeScript | **~5.7**, compilado con Vite |
| Toolchain frontend | Node.js | **18 o superior** + npm |

---

## 3. Frameworks y librerías principales

### Backend (Python)

| Área | Tecnología |
| --- | --- |
| API REST + WebSockets | **FastAPI** + **Uvicorn** |
| Concurrencia CPU | `concurrent.futures.ThreadPoolExecutor` |
| Sincronización | `queue.Queue`, `threading.Lock`, `threading.Semaphore` |
| Procesamiento de imagen | **Pillow**, **OpenCV**, **NumPy** |
| Backend GPU NVIDIA | **Numba** (kernels CUDA) |
| Backend GPU AMD / Intel | **PyOpenCL** (kernels OpenCL) |
| Calidad de código | **pylint**, **pytest** |
| Túnel para Colab | **pyngrok** |

### Frontend (dashboard web)

| Área | Tecnología |
| --- | --- |
| Framework UI | **React 18.3** + **Vite 6** + **TypeScript** |
| Estilos | **TailwindCSS 3.4** + componentes estilo **shadcn/ui** (Radix UI) |
| Gráficos | **Recharts** (gráfico de speedup CPU vs GPU) |
| Íconos | lucide-react |
| Linting | ESLint 9 + typescript-eslint |

---

## 4. Arquitectura 🏗️

El sistema utiliza un diseño estructurado bajo patrones de concurrencia, organizado en 5 capas:

```text
[ Capa 1: Carga / Lectura ] -> Escanea directorios y encola paths rápidamente
        |
        v
[ Capa 2: Cola de Entrada ] -> Estructura thread-safe (Productor-Consumidor)
        |
        v
[ Capa 3: Workers ] ---------> Pool de hilos CPU + Kernels GPU
        |      \
        |       \-----------> [ Cola de E/S ] -> [ Hilo de Guardado en Disco ]
        v
[ Capa 4: Agregador ] -------> Recolector de métricas y cálculo de Speedup
        |
        v
[ Capa 5: Dashboard ] -------> API FastAPI + Frontend React (progreso por WebSocket)
```

### Mapeo de capas a módulos

| Capa | Responsabilidad | Módulos |
| --- | --- | --- |
| 1 · Carga | Escanea la carpeta y encola paths | `pipeline/image_loader.py` |
| 2 · Cola | Buffer thread-safe productor-consumidor | `core/queue_manager.py` |
| 3 · Workers | Pool de hilos CPU + backend GPU + batching | `pipeline/worker.py`, `core/backend/`, `pipeline/gpu_batcher.py` |
| 4 · Agregador | Métricas, speedup y reporte CSV | `pipeline/result_aggregator.py`, `core/metrics.py`, `pipeline/report_exporter.py` |
| 5 · Dashboard | API REST/WebSocket + SPA en tiempo real | `api/`, `frontend/` |

### Patrón Strategy para detección de backend

Para lograr la máxima flexibilidad de hardware, el backend implementa el **patrón de diseño
Strategy**. Al inicializar el sistema se detecta el hardware disponible y se instancia
dinámicamente el backend correspondiente (`CUDABackend`, `OpenCLBackend` o `CPUBackend`). Todos
respetan la misma interfaz base `GPUBackend` con el método de entrada `process(image, operation)`.
De esta forma, el resto del pipeline (workers y colas) funciona de manera **agnóstica al hardware
subyacente**. La detección vive en `core/backend/factory.py` (`get_backend()`) y las estrategias
en `core/backend/cuda.py`, `opencl.py` y `cpu.py`.

---

## 5. Restricciones de hardware

El sistema tiene **detección automática de hardware** y utiliza el mejor disponible según la
siguiente prioridad:

| Hardware | Backend | Librería | Prioridad |
| --- | --- | --- | --- |
| NVIDIA GPU | CUDA | Numba | 1 (más alta) |
| AMD / Intel GPU | OpenCL | PyOpenCL | 2 |
| Cualquier CPU | Multithreading | `concurrent.futures` | 3 (fallback universal) |

- **No hay requisito duro de GPU:** si no se detecta ninguna, el sistema cae automáticamente al
  backend **CPU** (ThreadPoolExecutor), que funciona en cualquier máquina.
- **NVIDIA (CUDA):** requiere drivers NVIDIA actualizados y el **CUDA Toolkit** instalado.
- **AMD / Intel (OpenCL):** requiere drivers OpenCL compatibles (incluye GPUs integradas).
- **Forzar CPU (saltear OpenCL):** en GPUs integradas (p. ej. Ryzen) OpenCL puede saturar el
  procesamiento. Para evitar OpenCL y usar CPU, definir la variable de entorno
  `PARALLELVISION_DISABLE_OPENCL=1` antes de correr `python main.py` o la API. Solo afecta a
  OpenCL; la ruta CUDA se mantiene. Para volver al comportamiento normal, no definir la variable.
- **Control de saturación de GPU:** el envío a GPU está limitado por semáforo
  (`MAX_GPU_CONCURRENT_BATCHES = 2`) y trabaja por lotes (`MAX_GPU_BATCH_SIZE = 32`). Ante un
  error de memoria de video (OOM) el sistema hace **fallback automático a CPU** por imagen.
- **Entorno GPU alternativo — Google Colab (NVIDIA T4):** se puede correr el backend CUDA en una
  GPU dedicada de Colab exponiéndolo por túnel ngrok. Ver la guía en
  [docs/COLAB_GPU.md](docs/COLAB_GPU.md).

---

## 6. Sistema operativo y navegador

### Sistema operativo

Multiplataforma. Probado en:

- **Linux** (distribuciones modernas) y **macOS**.
- **Windows 10 / 11.** El path CUDA en Windows requiere drivers NVIDIA + CUDA Toolkit; el sistema
  localiza automáticamente las DLLs del Toolkit para Numba (`core/backend/cuda.py`).

El backend CPU y el dashboard funcionan en cualquiera de estos sistemas; la aceleración por GPU
depende únicamente de los drivers instalados.

### Navegador

El dashboard es una SPA web que se abre en **cualquier navegador moderno con soporte de
WebSocket** — Chrome, Edge o Firefox recientes. No hay una versión mínima fijada. En desarrollo
el dashboard se sirve en `http://localhost:5173`.

---

## 7. Puesta en marcha ⚙️

### Requisitos previos

- Python **3.11+**
- Node.js **18+** y npm (solo para el dashboard)
- git

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/UNLAM-PROG-C/2026-PROGC-Q1-M6.git
cd 2026-PROGC-Q1-M6/TP-Integrador

# 2. Crear y activar el entorno virtual
python -m venv venv
source venv/bin/activate        # Linux / macOS
# .\venv\Scripts\activate       # Windows (PowerShell)

# 3. Instalar dependencias del backend
pip install -r requirements.txt
```

### Ejecución — Modo CLI (headless)

Procesa una carpeta desde la línea de comandos, sin interfaz gráfica:

```bash
python main.py --input-dir ruta/a/imagenes --operation blur
```

Flags principales: `--operation` (`grayscale` · `edges` · `blur` · `equalize`), `--workers`,
`--queue-size`, `--output-dir`, `--report archivo.csv`, `--no-save`. Ver `python main.py --help`.

### Ejecución — Modo Dashboard (web)

La interfaz gráfica funciona con una arquitectura cliente-servidor (backend FastAPI + frontend
React/Vite).

```bash
# Terminal 1 — Backend (API + WebSockets) en http://localhost:8000
python -m uvicorn api.main:app --reload

# Terminal 2 — Frontend (dashboard) en http://localhost:5173
cd frontend
npm install
npm run dev
```

En desarrollo, Vite hace de **proxy** de `/api` y `/ws` hacia `http://localhost:8000`
(`frontend/vite.config.ts`), por lo que no hace falta configurar CORS.

> **Opcional — imágenes de prueba:** `python scripts/descargar_imagenes.py -n 100 -d public/images/in`
> descarga imágenes aleatorias para probar el sistema.
>
> 📖 **El uso guiado de la interfaz (paso a paso con capturas) está en el _Manual de Usuario_,
> que se entrega como documento aparte.**

---

## 8. Documentación relacionada

- [docs/DESCRIPCION_PARALLEL_VISION.md](docs/DESCRIPCION_PARALLEL_VISION.md) — propuesta y
  descripción técnica completa del proyecto.
- [docs/DECISION_GUI_REACT.md](docs/DECISION_GUI_REACT.md) — decisión de arquitectura: dashboard
  web (React + FastAPI) en lugar de escritorio (Tkinter/PyQt).
- [docs/COLAB_GPU.md](docs/COLAB_GPU.md) — guía para correr el backend CUDA en Google Colab (T4).
- [docs/GIT_STRATEGY.md](docs/GIT_STRATEGY.md) — flujo de trabajo git del equipo.
- [docs/PYTHON_STYLE_GUIDE.md](docs/PYTHON_STYLE_GUIDE.md) — guía de estilo de código Python.
- **Manual de Usuario** — documento aparte (uso guiado con capturas / video).
