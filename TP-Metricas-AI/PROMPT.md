# Prompt técnico — TP1 Parte 1 (Métricas de IA)

> **Materia:** Programación Concurrente — UNLAM — 1° Cuatrimestre 2026
> **Origen funcional:** subconjunto de la Fase 1 de **ParallelVision** (TP-Integrador)
> **Issues cubiertos:** #5, #6, #8, #10, #16, #17 (parcial: aggregator sin CSV) y #9 (solo rama CPU)

---

## LENGUAJE DE PROGRAMACIÓN

- Utilizar **Python 3.11+** con anotaciones de tipo, siguiendo el Google Python Style Guide adaptado a la cátedra.

## PLATAFORMA

- Desarrollar una aplicación **CPU-only multihilo** ejecutable en Linux/macOS/Windows, distribuida como paquete Python (`TP-Integrador/`), invocable por línea de comandos vía `python -m TP-Integrador.main`.
- Dependencias permitidas (únicas, declaradas en `requirements.txt`): `pillow`, `opencv-python`, `numpy`, `pylint`, `pytest`. Toda la concurrencia se resuelve con la **biblioteca estándar** (`threading`, `queue`, `concurrent.futures`).

## FUNCIONALIDAD A IMPLEMENTAR

- Desarrollar un **pipeline concurrente de procesamiento masivo de imágenes** que recorra una carpeta de entrada, distribuya el trabajo entre un pool de workers, aplique una operación de transformación a cada imagen y consolide métricas de rendimiento por operación de manera **thread-safe**, siguiendo el patrón **productor-consumidor** con cola acotada y agregador dedicado.

## REQUISITOS ESPECÍFICOS

- Se debe implementar la siguiente **estructura de paquete**, sin agregar archivos extra:

  ```text
  TP-Integrador/
  ├── core/
  │   ├── __init__.py
  │   ├── backend.py            # CPUBackend + get_backend() (solo CPU)
  │   ├── queue_manager.py      # ImageQueue
  │   └── metrics.py            # MetricsCollector
  ├── pipeline/
  │   ├── __init__.py
  │   ├── image_loader.py       # productor: escanea carpeta, encola paths
  │   ├── worker.py             # ProcessingWorker + ThreadPoolExecutor
  │   └── result_aggregator.py  # consumidor de resultados → MetricsCollector
  └── main.py                   # arma el pipeline, lo corre, imprime resumen
  ```

- Se debe implementar una clase `ImageQueue` en `core/queue_manager.py` con constructor `ImageQueue(maxsize: int)` y los métodos públicos `put(item)`, `get()`, `task_done()`, `join()`, `empty()` y `qsize()`. Internamente debe envolver `queue.Queue` sin exponer la cola interna.
- Se debe implementar una clase `CPUBackend` en `core/backend.py` con un único método público `process(image: np.ndarray, operation: str) -> np.ndarray` que soporte exactamente cuatro operaciones:

  | Operación   | Implementación esperada                          |
  |-------------|--------------------------------------------------|
  | `grayscale` | Conversión a escala de grises (OpenCV o Pillow). |
  | `edges`     | Detección de bordes con Canny (OpenCV).          |
  | `blur`      | Gaussian Blur con kernel configurable (OpenCV).  |
  | `equalize`  | Ecualización de histograma (OpenCV).             |

- Se debe implementar la función `get_backend() -> CPUBackend` en `core/backend.py` que por ahora retorna siempre `CPUBackend()`, dejando un comentario `# TODO: detección CUDA/OpenCL en futura iteración`.
- Se debe implementar la clase `MetricsCollector` en `core/metrics.py` con los métodos `record(image_name: str, operation: str, elapsed_ms: float)` (protegido con `threading.Lock`), `get_summary() -> dict` (devolviendo por operación: `count`, `avg_ms`, `min_ms`, `max_ms`) y `get_throughput(total_seconds: float) -> float` (imágenes/segundo global).
- Se debe implementar el módulo `pipeline/image_loader.py` con la función `scan_folder(input_dir: str) -> list[str]` que filtre por extensiones válidas (constante `IMAGE_EXTENSIONS`), y la función `enqueue_paths(paths: list[str], queue: ImageQueue, num_workers: int)` que encole los paths y publique un sentinela `None` por cada worker.
- Se debe implementar la clase `ProcessingWorker` en `pipeline/worker.py` que reciba en su constructor `input_queue`, `result_queue`, `backend` y `operation`. Su método `run()` debe consumir hasta sentinela `None`, medir el tiempo con `time.perf_counter()` antes y después de `backend.process()`, y encolar la tupla `(image_name, operation, elapsed_ms)` en la cola de resultados. Los workers deben lanzarse vía `concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)`.
- Se debe implementar el módulo `pipeline/result_aggregator.py` como un hilo dedicado que consuma la cola de resultados hasta recibir un sentinela `None` y llame a `MetricsCollector.record()` por cada resultado.
- Se debe implementar `main.py` que parsee los argumentos `--input-dir`, `--operation`, `--workers` (default 4) y `--queue-size` (default 32), arme el pipeline, lo arranque, espere su finalización e imprima el resumen formateado.
- Se debe garantizar **shutdown limpio**: workers y agregador deben drenar las colas y finalizar sin colgarse, usando sentinelas `None` o `queue.join()` (justificando la elección con un comentario breve). El programa debe llamar a `executor.shutdown(wait=True)`.
- Se debe usar `concurrent.futures.ThreadPoolExecutor` con número de workers configurable mediante constante `MAX_WORKER_THREADS`.
- Se debe usar `queue.Queue` thread-safe acotada (`maxsize` configurable mediante constante `DEFAULT_QUEUE_SIZE`) para la cola de entrada y la cola de resultados.
- Se debe usar `threading.Lock` para proteger el estado interno de `MetricsCollector`.
- El sistema debe imprimir, al finalizar, un resumen con el siguiente formato:

  ```text
  [INFO] Backend activo: CPUBackend
  [INFO] Procesadas 50 imágenes en 3.21 s (15.58 img/s)

  Resumen por operación:
    grayscale  →  count=50  avg=64.2 ms  min=41.0 ms  max=128.5 ms
  ```

- Se debe usar `logging.info('msg: %s', var)` para todas las trazas. **Nunca f-strings dentro de logging.**
- Se debe levantar `ValueError` en `CPUBackend.process()` si la operación recibida no pertenece al conjunto `VALID_OPERATIONS`.

## RESTRICCIONES

- **Máximo 15 líneas por función o método** (sin contar firma ni docstring). Esta regla es de la cátedra y no admite excepciones.
- **Sin números mágicos.** Toda constante numérica o string fijo debe declararse como `CAPS_WITH_UNDER` al tope del módulo. Constantes obligatorias mínimas: `IMAGE_EXTENSIONS`, `VALID_OPERATIONS`, `MAX_WORKER_THREADS`, `DEFAULT_QUEUE_SIZE`, `GAUSSIAN_KERNEL_SIZE`, `CANNY_LOW_THRESHOLD`, `CANNY_HIGH_THRESHOLD`.
- **Línea máxima: 80 caracteres.** Sin backslash de continuación; usar paréntesis. Indentación: 4 espacios. Sin punto y coma al final de línea.
- **Imports absolutos** (`from parallelvision.core import ...`), nunca relativos. Orden: `__future__` → stdlib → terceros → proyecto, separados por línea en blanco.
- **Type hints obligatorios** en toda la API pública. **Docstrings estilo Google** (`Args:` / `Returns:` / `Raises:`) en toda función pública o de más de 10 líneas.
- **Mutable defaults prohibidos.** Usar `None` y asignar dentro de la función.
- **Excepciones específicas.** Nunca `except:` ni `except Exception: pass`.
- **Nomenclatura obligatoria:** clases en `CapWords`; funciones, métodos y variables en `lower_with_under`; constantes en `CAPS_WITH_UNDER`; identificadores privados con prefijo `_`.
- El código debe pasar `pylint` con `max-line-length=80` y `max-module-lines=200` sin warnings no justificados.
- **Concurrencia obligatoria con la biblioteca estándar.** Está prohibido usar `multiprocessing`, `numba`, `pyopencl`, frameworks de GUI, ni librerías de paralelismo de alto nivel (`joblib`, `dask`, `ray`, `asyncio`, etc.).
- Toda comunicación entre hilos debe pasar por `queue.Queue`. Todo acceso a estado compartido debe pasar por `threading.Lock`. **No asumir atomicidad de built-ins.**
- **Sin race conditions:** ninguna operación read-modify-write sobre estado compartido fuera del `Lock`.
- **Sin deadlocks:** orden de adquisición de recursos consistente; los sentinelas deben garantizar que todos los hilos finalicen.
- **Sin busy-wait:** está prohibido cualquier bucle de polling con `sleep`. Usar `Queue.get()` bloqueante o `Event.wait()`.
- **Exclusión mutua estricta:** sólo un hilo puede modificar el dict de métricas a la vez.
- **NO se debe generar:** backends GPU (CUDA, OpenCL, Numba), detección dinámica de hardware, GUI / dashboard / matplotlib, exportación a CSV, tests unitarios, ni manejo de configuración por archivo (YAML, TOML, JSON).
- La salida de la generación debe ser **un único bloque de código por archivo**, en el orden de la estructura de paquete declarada arriba, precedido por el path relativo del archivo. Sin explicaciones intercaladas. Al final, un único párrafo de máximo 5 líneas describiendo cómo se garantiza la ausencia de deadlocks y por qué se eligió el mecanismo de finalización (sentinela vs `join`).
