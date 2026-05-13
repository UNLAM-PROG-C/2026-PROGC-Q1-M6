# ParallelVision

## Pipeline de Procesamiento de Imágenes con CPU + GPU

---

**Universidad Nacional de La Matanza**  
Programación Concurrente – 1° Cuatrimestre 2026  
Trabajo Práctico Integrador – Propuesta

---

## Integrantes

| Nombre | DNI |
| --- | --- |
| Felice, Tomás Agustín | 44.789.809 |
| De La Cruz Zamudio, Axel Nahuel | 41.063.583 |
| Graneros, Brian Ariel | 41.130.084 |
| Sanchez, Kevin Erik | 41.173.649 |

---

## 1. Descripción general

**ParallelVision** es una herramienta de escritorio para el procesamiento masivo y paralelo de imágenes. El usuario selecciona una carpeta con decenas o cientos de archivos, elige las transformaciones a aplicar (escala de grises, detección de bordes, blur gaussiano, ecualización de histograma, entre otras) y el sistema las ejecuta en paralelo, aprovechando simultáneamente los núcleos de la CPU mediante un pool de hilos y los núcleos de la GPU a través de kernels de cómputo paralelo.

Una característica central del diseño es la **detección automática de backend en runtime**: al iniciar, la aplicación detecta qué hardware GPU está disponible en la máquina y selecciona automáticamente el mejor motor de procesamiento sin que el usuario tenga que configurar nada. Esto hace que el aplicativo sea compatible con cualquier GPU del mercado — NVIDIA, AMD e Intel — y funcione en todas las máquinas del equipo sin cambiar una línea de código.

El resultado es un dashboard en tiempo real que muestra el progreso imagen por imagen, el tiempo acumulado y una comparativa cuantitativa de rendimiento CPU vs GPU, ilustrando de forma tangible el beneficio del paralelismo masivo.

El proyecto tiene valor de mercado real: estudios fotográficos, equipos de machine learning y creadores de contenido necesitan procesar grandes volúmenes de imágenes de forma eficiente. Herramientas similares como Adobe Lightroom o Topaz Photo AI cuestan cientos de dólares por año; ParallelVision apunta a ser una alternativa liviana, open-source y compatible con cualquier hardware.

---

## 2. Objetivos

### Objetivo general

Desarrollar una aplicación funcional que demuestre de forma práctica y medible los beneficios de la programación concurrente y paralela aplicada a un caso de uso real, con soporte transparente para múltiples tecnologías de GPU.

### Objetivos específicos

- Implementar un pipeline productor-consumidor usando colas sincronizadas para gestionar el flujo de imágenes.
- Aplicar un pool de hilos (`ThreadPoolExecutor`) para el procesamiento paralelo en CPU.
- Diseñar un sistema de detección automática de backend GPU en runtime (CUDA → OpenCL → CPU), implementando el patrón de diseño Strategy.
- Desarrollar kernels de procesamiento para CUDA (Numba) y OpenCL (PyOpenCL), intercambiables a través de una interfaz común.
- Garantizar escritura segura de resultados mediante mutex y locks, evitando condiciones de carrera.
- Medir y visualizar el speedup obtenido (CPU vs GPU) con métricas cuantitativas.
- Construir una interfaz gráfica que muestre el progreso y los resultados en tiempo real.

---

## 3. Conceptos de programación concurrente aplicados

| Concepto | Aplicación en ParallelVision |
| --- | --- |
| **Hilos (Threads)** | Pool de hilos para procesar imágenes en paralelo en la CPU. Cada hilo toma un trabajo de la cola y aplica las transformaciones de forma independiente. |
| **Productor-Consumidor** | El módulo de carga actúa como productor (agrega imágenes a la queue); los workers CPU/GPU actúan como consumidores. |
| **Mutex / Lock** | El agregador de resultados usa un lock para escritura segura en la estructura compartida de métricas, evitando data races. |
| **Semáforos** | Limitan la cantidad de imágenes enviadas a la GPU simultáneamente, evitando saturación de memoria de video. |
| **Cola sincronizada (Queue)** | Estructura thread-safe que desacopla la carga del procesamiento, permitiendo que ambas etapas corran en paralelo. |
| **Paralelismo masivo (GPU)** | Kernels CUDA u OpenCL que ejecutan la misma operación sobre miles de píxeles de forma verdaderamente simultánea. |
| **Sincronización CPU-GPU** | Manejo explícito de transferencia de datos entre RAM y memoria de video (host-to-device / device-to-host) con sincronización de streams. |

---

## 4. Arquitectura del sistema

El sistema se organiza en dos grandes bloques: la **detección de backend** que ocurre al inicio, y el **pipeline unificado** que opera igual independientemente del hardware disponible.

### 4.1 Detección automática de backend (Strategy Pattern)

Al arrancar, la aplicación ejecuta la siguiente lógica de detección en orden de prioridad:

```
App arranca
    │
    ▼
¿GPU NVIDIA disponible?  ──Sí──▶  Backend CUDA    (Numba CUDA kernels)
    │ No                                │
    ▼                                   │
¿GPU OpenCL disponible?  ──Sí──▶  Backend OpenCL  (PyOpenCL kernels)
    │ No                                │
    ▼                                   │
Backend CPU fallback     ◀─────────────┘
(ThreadPoolExecutor)
    │
    ▼
Pipeline unificado — Queue + Lock + Dashboard
(idéntico para todos los backends)
```

La detección se implementa mediante el patrón **Strategy**: todos los backends exponen la misma interfaz (`process(image, operation)`), por lo que el resto de la aplicación nunca interactúa directamente con CUDA ni OpenCL. Esto desacopla completamente la lógica de negocio del hardware subyacente.

```python
class GPUBackend:
    """Interfaz común para todos los backends."""
    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        raise NotImplementedError

class CUDABackend(GPUBackend):    # GPU NVIDIA
    def process(self, image, operation): ...  # Numba CUDA

class OpenCLBackend(GPUBackend):  # GPU AMD / Intel integrada
    def process(self, image, operation): ...  # PyOpenCL

class CPUBackend(GPUBackend):     # Fallback sin GPU
    def process(self, image, operation): ...  # Pillow / OpenCV

def get_backend() -> GPUBackend:
    try:
        import numba.cuda
        if numba.cuda.is_available():
            return CUDABackend()
    except: pass
    try:
        import pyopencl as cl
        if cl.get_platforms():
            return OpenCLBackend()
    except: pass
    return CPUBackend()
```

### 4.2 Pipeline unificado

Una vez seleccionado el backend, el flujo de datos es el mismo en todos los casos:

```
┌─────────────────────────────────────────┐
│          Carga de imágenes              │  Capa 1
│  Escanea carpeta → encola paths         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Cola de trabajo (Queue)         │  Capa 2
│     thread-safe · productor-consumidor  │
└───────────┬─────────────────────┬───────┘
            │                     │
            ▼                     ▼
┌───────────────────┐   ┌──────────────────────┐
│  Pool de hilos    │   │  Backend GPU activo   │  Capa 3
│  CPU              │   │  CUDA / OpenCL        │
│  ThreadPoolExec.  │   │  (según hardware)     │
└───────────┬───────┘   └──────────┬───────────┘
            │                      │
            └──────────┬───────────┘
                       ▼
┌─────────────────────────────────────────┐
│       Agregador de resultados           │  Capa 4
│   threading.Lock · métricas · speedup   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│        Dashboard en tiempo real         │  Capa 5
│    Tkinter / PyQt · matplotlib          │
└─────────────────────────────────────────┘
```

---

## 5. Tecnologías y herramientas

| Área | Tecnología | Uso específico |
| --- | --- | --- |
| Lenguaje principal | Python 3.11+ | Toda la lógica de la aplicación |
| Concurrencia CPU | `concurrent.futures` – `ThreadPoolExecutor` | Pool de hilos para procesamiento paralelo en CPU |
| Procesamiento imagen CPU | Pillow, OpenCV | Transformaciones: blur, bordes, escala de grises |
| Backend GPU NVIDIA | Numba (CUDA) | Kernels paralelos en GPU NVIDIA; compatible con Google Colab |
| Backend GPU AMD / Intel | PyOpenCL | Kernels paralelos en GPUs con soporte OpenCL (incluye integradas) |
| Detección de backend | Lógica de runtime propia | Strategy Pattern — selección automática sin intervención del usuario |
| Cola sincronizada | `queue.Queue` | Buffer thread-safe entre productor y consumidores |
| Sincronización | `threading.Lock` / `Semaphore` | Protección de recursos compartidos |
| Interfaz gráfica | Tkinter o PyQt5 | Dashboard con progreso en tiempo real |
| Métricas / gráficos | matplotlib | Gráfico de speedup CPU vs GPU en vivo |
| Entorno alternativo | Google Colab (GPU T4/A100) | Demo del backend CUDA con hardware dedicado y speedups más altos |

### Compatibilidad de hardware por backend

| Hardware | Backend activado | Tecnología |
| --- | --- | --- |
| GPU NVIDIA (GTX, RTX, Tesla) | CUDA | Numba CUDA |
| GPU AMD dedicada o integrada | OpenCL | PyOpenCL |
| GPU Intel integrada (Ryzen, Core) | OpenCL | PyOpenCL |
| Sin GPU / CPU only | CPU fallback | ThreadPoolExecutor |
| Google Colab | CUDA (T4 / A100) | Numba CUDA |

---

## 6. Funcionalidades del aplicativo

### Núcleo del procesamiento

- Selección de carpeta de entrada y carpeta de salida desde la interfaz.
- Elección de transformaciones: escala de grises, detección de bordes (Canny), blur gaussiano, ecualización de histograma.
- Detección automática del backend óptimo al iniciar (CUDA → OpenCL → CPU).
- Indicador visible en la interfaz del backend activo y el hardware detectado.
- Cola de trabajo configurable: el usuario ajusta el tamaño del pool de hilos y la cantidad de lotes GPU.

### Dashboard en tiempo real

- Barra de progreso global y por imagen actualmente en procesamiento.
- Velocidad de procesamiento en imágenes/segundo y tiempo transcurrido.
- Estimación de tiempo restante basada en el rendimiento actual.
- Gráfico en vivo de speedup: tiempo CPU vs tiempo GPU por lote de imágenes.

### Reporte final

- Tabla comparativa de tiempos por operación (CPU vs GPU).
- Speedup promedio expresado como factor (ej: *"GPU fue 8.4x más rápida"*).
- Exportación del reporte como CSV con todos los datos de la sesión.

---

## 7. Valor de mercado y diferencial competitivo

### Segmentos de usuario objetivo

- **Fotógrafos y estudios fotográficos:** necesitan exportar y convertir cientos de fotos por sesión.
- **Equipos de machine learning y data science:** preprocesamiento masivo de datasets de imágenes.
- **Creadores de contenido digital:** redimensionado, conversión de formato y filtros en lote.
- **Investigadores académicos:** herramientas ligeras sin dependencia de servicios en la nube.

### Diferencial respecto a herramientas existentes

| Característica | ParallelVision | Alternativas (Lightroom, Topaz) |
| --- | --- | --- |
| Costo | Open-source / gratuito | USD 100–200/año |
| Compatibilidad GPU | NVIDIA + AMD + Intel (cualquier GPU) | Solo NVIDIA o propietario |
| Detección automática de hardware | Sí, sin configuración | No |
| Dashboard de speedup en tiempo real | Sí | No |
| Extensibilidad | Kernels propios en Python | Cerrado |
| Funcionamiento offline | 100% local | Algunos requieren nube |

---

## 8. Entregables previstos

- [ ] Código fuente completo en repositorio GitHub asignado al grupo.
- [ ] `README.md` con descripción, instrucciones de instalación y manual de usuario.
- [ ] Informe en formato PDF con carátula, descripción técnica, manual de usuario y conclusiones.
- [ ] Video de demostración del aplicativo en funcionamiento (enlace incluido en el informe).
- [ ] Cuaderno Google Colab demostrando el backend CUDA con GPU Tesla/A100 y métricas de speedup a mayor escala.
