# Conclusiones — ParallelVision

**Trabajo Práctico Integrador · Programación Concurrente · UNLAM · 1.º Cuatrimestre 2026**

---

## 1. Introducción

**ParallelVision** es un pipeline de procesamiento masivo de imágenes (escala de
grises, bordes, desenfoque y ecualización) que aprovecha CPU o GPU de forma
transparente. Se usa como **CLI headless** (`python main.py`) y como **dashboard
web** (FastAPI + React/Vite con progreso por WebSocket). Estas conclusiones
recogen los aprendizajes más relevantes, fundamentados en el código del
repositorio.

---

## 2. Detección automática de hardware

Desde el propio código se puede **interrogar a la máquina** para saber qué GPU
tiene —o si no tiene ninguna— y ofrecer aceleración de forma transparente. La
detección vive en [core/backend/factory.py](../core/backend/factory.py) y sigue
el orden **CUDA → OpenCL → CPU**, en capas cada vez más específicas:

1. **Import guard**: si la librería (`numba`, `pyopencl`) no está, un
   `except ImportError` descarta el backend sin intentar usarlo.
2. **Consulta al driver**: se verifica que exista hardware real
   (`cuda.is_available()`, `cl.get_platforms()`).
3. **Excepciones específicas**: ante un error conocido del driver se registra y
   se **degrada al siguiente backend** en lugar de abortar.

El resultado es que **el mismo código corre en cualquier máquina**, porque los
tres backends implementan el mismo contrato (`process()` / `process_batch()` en
[core/backend/base.py](../core/backend/base.py)). Para robustecer la detección
resolvimos casos de borde reales —un parche de rutas del CUDA Toolkit en Windows
y un *fallback* al "CUDA Simulator"—, lo que nos enseñó que detectar hardware es
tanto consultar APIs como manejar los entornos donde esas APIs fallan.

---

## 3. Impacto del diseño GPU en el rendimiento

**Tener GPU no garantiza velocidad: hay que saber aprovecharla.** Una
implementación ingenua puede ser más lenta que la CPU. Dos decisiones fueron
clave:

- **Grid 3D para batches**: los kernels batcheados usan un grid `(H, W, N)`
  (`_get_batch_grid_dims` en [core/backend/cuda.py](../core/backend/cuda.py)), de
  modo que **un solo lanzamiento procesa N imágenes** y llena de trabajo los
  miles de hilos del dispositivo, en vez de dejarlos ociosos.
- **Amortizar la transferencia**: el costo dominante no es calcular, sino **mover
  datos por el bus PCIe**. Por eso `process_batch()` apila las N imágenes y hace
  **una sola transferencia por lote**. El `GpuBatcher`
  ([pipeline/gpu_batcher.py](../pipeline/gpu_batcher.py)) acumula imágenes del
  mismo shape y dispara el lote al llenarse.

La lección central: si por cada imagen hacés un lanzamiento de kernel y un par de
transferencias, el overhead se come la ventaja de la GPU. Suman también el
`warmup()` (descuenta la compilación JIT de las métricas), las sumas atómicas
(`cuda.atomic.add`) para el histograma y la ecualización con un único viaje D2H.

---

## 4. Concurrencia

El pipeline nos obligó a usar varios mecanismos distintos, cada uno para un
problema puntual:

- **Cola acotada** (`ImageQueue`, [core/queue_manager.py](../core/queue_manager.py)):
  el `maxsize` da **contrapresión** natural —si los consumidores no dan abasto, el
  productor se bloquea en vez de cargar toda la memoria.
- **Semáforo** (`_gpu_semaphore` en [core/backend/base.py](../core/backend/base.py)):
  limita a 2 los lotes GPU simultáneos para **no saturar la VRAM**; regula un
  recurso limitado, no es un lock binario.
- **Lock** (`MetricsCollector`, [core/metrics.py](../core/metrics.py)): protege el
  estado escrito por muchos hilos; las lecturas **copian dentro del lock y
  calculan afuera** para retenerlo lo mínimo.
- **Thread-Local Storage** (`_COMPUTE_TLS`, `_FALLBACK_TLS` en
  [pipeline/worker.py](../pipeline/worker.py)): estado por hilo sin sincronizar.
  En vez de proteger lo compartido, **eliminamos el hecho de compartirlo**.
- **Sentinela / *poison pill***: el `ResultAggregator`
  ([pipeline/result_aggregator.py](../pipeline/result_aggregator.py)) termina al
  recibir `None`; con `task_done()`/`join()` logra un **apagado ordenado**.
- **Hilo de E/S dedicado** (`ImageSaver`): desacopla el guardado a disco para que
  los workers de cómputo no se frenen esperando al disco.

La conclusión: **no existe "la" primitiva de concurrencia**; la habilidad está en
elegir la adecuada (cola, semáforo, lock, TLS, sentinela) para cada problema.

---

## 5. Trabajo coordinado en equipo

Buena parte del aprendizaje fue de **proceso**:

- **Estándares escritos**: una guía de estilo común
  ([docs/PYTHON_STYLE_GUIDE.md](PYTHON_STYLE_GUIDE.md)) con reglas duras (máximo
  15 líneas por función, sin números mágicos, docstrings) volvió objetivas las
  revisiones.
- **Git disciplinado**: GitHub Flow con `develop` como rama de integración, **PRs
  con revisión aprobada** y Conventional Commits ([docs/GIT_STRATEGY.md](GIT_STRATEGY.md)),
  lo que nos permitió **verificarnos mutuamente** el código.
- **Arquitectura temprana**: definir las **5 capas** y el contrato `GPUBackend`
  dio interfaces estables para avanzar en paralelo sin bloquearnos.

La coordinación no se improvisa: los estándares, los PRs y la arquitectura
acordada al inicio fueron la infraestructura del trabajo en paralelo.

---

## 6. Patrones de diseño

Aplicamos patrones resolviendo problemas reales: **Strategy** (`GPUBackend` con
CPU/CUDA/OpenCL intercambiables), **Factory** (`get_backend()` encapsula la
detección) y **Producer–Consumer** (colas thread-safe entre capas). Dos
consecuencias valiosas:

1. **Fallback resiliente en caliente**: si un lote agota la VRAM, `process()` no
   falla —captura la excepción, degrada esa imagen a CPU (`_handle_oom`) y la
   etiqueta como `cpu-fallback`. La abstracción de backend lo hizo trivial.
2. **Extensibilidad demostrada**: agregar OpenCL **no requirió tocar el
   pipeline**, solo implementar la interfaz. Es la prueba concreta del valor de
   Strategy.

El sistema se cierra con una **métrica de *speedup* GPU/CPU en vivo**
(`_speedup_from_records` en [core/metrics.py](../core/metrics.py)) sobre un
dashboard en tiempo real: no solo aceleramos, construimos la instrumentación para
**medir y validar empíricamente** la aceleración.

---

## 7. Conclusión general

Tres pilares resumen el trabajo:

1. **Concurrencia bien modelada** — la primitiva correcta para cada problema.
2. **GPU bien aprovechada** — el hardware paralelo solo rinde si se lo alimenta
   bien (batching, grid 3D, transferencias amortizadas).
3. **Proceso de equipo disciplinado** — estándares, revisión por PRs y
   arquitectura acordada de antemano.

El aprendizaje central: **la performance no es un accidente del hardware, sino
una consecuencia del diseño**. La misma GPU puede acelerar diez veces o frenar el
sistema según cómo se la use.
