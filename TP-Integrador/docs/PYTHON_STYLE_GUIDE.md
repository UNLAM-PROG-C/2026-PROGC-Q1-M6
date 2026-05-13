# Google Python Style Guide — Referencia del Proyecto

> **ParallelVision · Programación Concurrente · UNLAM 2026**
> Este documento es una adaptación de la [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) oficial, incorporada como estándar de codificación del proyecto. Todo el código del repositorio debe adherir a estas convenciones.

---

## Índice

1. [Herramientas de análisis estático](#1-herramientas-de-análisis-estático)
2. [Reglas del lenguaje Python](#2-reglas-del-lenguaje-python)
   - [2.1 Imports](#21-imports)
   - [2.2 Excepciones](#22-excepciones)
   - [2.3 Estado global mutable](#23-estado-global-mutable)
   - [2.4 Comprensiones y generadores](#24-comprensiones-y-generadores)
   - [2.5 Iteradores y operadores por defecto](#25-iteradores-y-operadores-por-defecto)
   - [2.6 Funciones lambda](#26-funciones-lambda)
   - [2.7 Expresiones condicionales](#27-expresiones-condicionales)
   - [2.8 Valores de argumento por defecto](#28-valores-de-argumento-por-defecto)
   - [2.9 Evaluación True/False](#29-evaluación-truefalse)
   - [2.10 Threading](#210-threading)
   - [2.11 Anotaciones de tipo](#211-anotaciones-de-tipo)
3. [Reglas de estilo Python](#3-reglas-de-estilo-python)
   - [3.1 Punto y coma](#31-punto-y-coma)
   - [3.2 Longitud de línea](#32-longitud-de-línea)
   - [3.3 Paréntesis](#33-paréntesis)
   - [3.4 Indentación](#34-indentación)
   - [3.5 Líneas en blanco](#35-líneas-en-blanco)
   - [3.6 Espacios en blanco](#36-espacios-en-blanco)
   - [3.7 Comentarios y docstrings](#37-comentarios-y-docstrings)
   - [3.8 Strings](#38-strings)
   - [3.9 Archivos y recursos](#39-archivos-y-recursos)
   - [3.10 Comentarios TODO](#310-comentarios-todo)
   - [3.11 Orden de imports](#311-orden-de-imports)
   - [3.12 Nomenclatura (Naming)](#312-nomenclatura-naming)
   - [3.13 Función main](#313-función-main)
   - [3.14 Longitud de funciones](#314-longitud-de-funciones)
   - [3.15 Anotaciones de tipo — detalles](#315-anotaciones-de-tipo--detalles)
4. [Palabra final](#4-palabra-final)

---

## 1. Herramientas de análisis estático

El proyecto utiliza **`pylint`** para detección de errores y problemas de estilo.

```bash
# Ejecutar sobre un módulo
pylint parallelvision/core/backend.py

# Listar todos los warnings disponibles
pylint --list-msgs

# Ver detalle de un warning específico
pylint --help-msg=invalid-name
```

### Supresión de warnings

Cuando un warning no aplica, se suprime a nivel de línea con un comentario explicativo:

```python
def do_PUT(self):  # WSGI name, so pylint: disable=invalid-name
    ...
```

Los argumentos no utilizados se eliminan explícitamente al inicio de la función:

```python
def process_image(image: np.ndarray, operation: str, flags: int | None = None) -> np.ndarray:
    del flags  # Unused in this backend implementation.
    ...
```

> **Regla del proyecto:** `pylint` debe ejecutarse antes de abrir un Pull Request. No se aceptan PRs con warnings no justificados.

---

## 2. Reglas del lenguaje Python

### 2.1 Imports

Usar `import` solo para paquetes y módulos, nunca para tipos, clases o funciones individuales (salvo excepciones del módulo `typing`).

```python
# ✅ Correcto
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Sequence

# ❌ Incorrecto — importar con nombre relativo
import backend         # ambiguo
from . import backend  # relativo no recomendado

# ✅ Siempre usar el path completo del paquete
from parallelvision.core import backend
from parallelvision.core.backend import GPUBackend
```

**Regla:** no usar imports relativos. Siempre usar el nombre completo del paquete.

---

### 2.2 Excepciones

Las excepciones están permitidas, pero deben usarse con cuidado.

```python
# ✅ Correcto: raise con tipo específico y mensaje descriptivo
def process(self, image: np.ndarray, operation: str) -> np.ndarray:
    if image is None:
        raise ValueError(f'image cannot be None, got: {image!r}')
    if operation not in VALID_OPERATIONS:
        raise ValueError(f'Unknown operation: {operation!r}. Valid: {VALID_OPERATIONS}')
    ...

# ✅ Correcto: capturar excepciones específicas
try:
    result = backend.process(image, operation)
except (IOError, MemoryError) as e:
    logging.error('Processing failed: %r', e)
    raise

# ❌ Incorrecto: catch-all oculta errores reales
try:
    result = backend.process(image, operation)
except:          # nunca
    pass

except Exception:  # evitar salvo re-raise o punto de aislamiento
    pass
```

**Reglas adicionales:**
- No usar `assert` en lugar de validaciones de precondición en código de producción.
- Minimizar el código dentro de un bloque `try/except`.
- Las excepciones propias del proyecto deben heredar de una clase existente y terminar en `Error` (ej: `BackendDetectionError`).
- Usar `finally` para limpieza de recursos.

---

### 2.3 Estado global mutable

Evitar el estado global mutable.

```python
# ✅ Permitido: constantes de módulo (inmutables)
MAX_GPU_BATCH_SIZE = 64
VALID_OPERATIONS = ('grayscale', 'edges', 'blur', 'equalize')
_DEFAULT_THREAD_COUNT = 4  # interna al módulo

# ❌ Evitar: estado mutable a nivel de módulo
_active_backend = None          # mutable — usar instancia de clase
_processed_images = []          # mutable — pasar como argumento
```

Si el estado global es inevitable, debe ser interno (prefijo `_`) y accedido únicamente a través de funciones o métodos públicos, con un comentario que justifique el diseño.

---

### 2.4 Comprensiones y generadores

Permitidas para casos simples. Priorizar la legibilidad sobre la concisión.

```python
# ✅ Correcto: comprensión simple y legible
image_paths = [path for path in folder.iterdir() if path.suffix in IMAGE_EXTENSIONS]

# ✅ Correcto: comprensión multilínea bien formateada
valid_results = [
    result
    for result in raw_results
    if result.elapsed_ms > 0
]

# ❌ Incorrecto: múltiples for/if en una sola comprensión
bad = [(x, y) for x in range(10) for y in range(5) if x * y > 10]

# ✅ Correcto: usar loop explícito para lógica compleja
result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))
```

---

### 2.5 Iteradores y operadores por defecto

Usar los iteradores por defecto de los tipos que los soportan.

```python
# ✅ Correcto
for path in image_paths: ...
for key, value in metrics.items(): ...
if operation in VALID_OPERATIONS: ...

# ❌ Incorrecto — innecesariamente explícito
for key in metrics.keys(): ...
for line in file.readlines(): ...
```

---

### 2.6 Funciones lambda

Permitidas para lógica de una sola línea. Si supera los 60–80 caracteres, definir como función.

```python
# ✅ Permitido: lambda simple
sort_by_time = sorted(results, key=lambda r: r.elapsed_ms)

# ✅ Preferir operator para operaciones matemáticas
import operator
product = reduce(operator.mul, values, 1)

# ❌ Lambda demasiado larga — definir función con nombre
process = lambda img, op, scale, rotate, flip: apply_all(img, op, scale, rotate, flip)
```

---

### 2.7 Expresiones condicionales (ternario)

Permitidas para casos simples donde cada parte cabe en una línea.

```python
# ✅ Correcto
backend_name = 'CUDA' if cuda_available else 'CPU'

# ✅ Correcto: split en múltiples líneas cuando es necesario
label = ('GPU activa' if self._backend_type == 'gpu'
         else 'CPU fallback')

# ❌ Incorrecto: demasiado largo para una expresión ternaria
mode = ('procesamiento paralelo con aceleración de hardware' if gpu_detected and driver_ok else 'procesamiento secuencial en CPU sin GPU disponible')
```

---

### 2.8 Valores de argumento por defecto

No usar objetos mutables como valores por defecto.

```python
# ✅ Correcto
def process_batch(images: list[np.ndarray], operations: Sequence[str] | None = None) -> list[np.ndarray]:
    if operations is None:
        operations = ['grayscale']
    ...

# ✅ Correcto: tupla vacía (inmutable)
def configure(operations: Sequence[str] = ()) -> None:
    ...

# ❌ Incorrecto: lista mutable como default
def process_batch(images, operations=[]):   # el default se comparte entre llamadas
    ...

# ❌ Incorrecto: valor evaluado en tiempo de importación
def stamp_result(result, ts=time.time()):   # ts no cambia entre llamadas
    ...
```

---

### 2.9 Evaluación True/False

Usar la evaluación implícita de falsedad cuando sea posible.

```python
# ✅ Correcto
if not image_queue:
    return
if results:
    export_csv(results)

# Verificar None siempre con `is None`
if backend is None:
    backend = get_backend()

# ✅ Correcto para enteros
if batch_size % 8 == 0:
    use_optimized_kernel()

# ❌ Incorrecto
if len(image_queue) == 0:   # usar `if not image_queue`
    return
if results != []:           # usar `if results`
    export_csv(results)
if backend == None:         # usar `is None`
    ...

# ⚠️ Especial para NumPy arrays
if not image.size:          # correcto para np.ndarray
    raise ValueError('Empty image array.')
```

---

### 2.10 Threading

No confiar en la atomicidad de los tipos built-in de Python.

```python
# ✅ Correcto: usar Queue para comunicación entre hilos
from queue import Queue

work_queue: Queue = Queue(maxsize=100)
results_queue: Queue = Queue()

# ✅ Correcto: usar Lock para proteger recursos compartidos
import threading

_metrics_lock = threading.Lock()
_metrics: list[dict] = []

def record_result(data: dict) -> None:
    with _metrics_lock:
        _metrics.append(data)

# ✅ Correcto: usar Semaphore para limitar acceso concurrente
_gpu_semaphore = threading.Semaphore(MAX_GPU_CONCURRENT_BATCHES)

def submit_to_gpu(batch: list[np.ndarray]) -> list[np.ndarray]:
    with _gpu_semaphore:
        return _active_backend.process_batch(batch)

# ❌ Incorrecto: asumir atomicidad de dict sin lock
_shared_dict['key'] = value   # no es atómico en todos los casos
```

---

### 2.11 Anotaciones de tipo

Se usan anotaciones de tipo en todo el código público del proyecto.

```python
# ✅ Correcto
def get_backend() -> GPUBackend:
    ...

def process(self, image: np.ndarray, operation: str) -> np.ndarray:
    ...

def record(self, image_name: str, backend: str, elapsed: float) -> None:
    ...

# Para variables con tipo difícil de inferir
active_backend: GPUBackend = get_backend()
```

---

## 3. Reglas de estilo Python

### 3.1 Punto y coma

**Nunca** terminar líneas con punto y coma. **Nunca** poner dos sentencias en una misma línea separadas por punto y coma.

```python
# ✅ Correcto
x = 1
y = 2

# ❌ Incorrecto
x = 1; y = 2
return result;
```

---

### 3.2 Longitud de línea

**Máximo 80 caracteres por línea.**

Excepciones: imports largos, URLs en comentarios, strings constantes sin espacios.

**No usar backslash para continuar líneas.** Usar paréntesis implícitos:

```python
# ✅ Correcto: continuación con paréntesis implícitos
result = process_image(
    image=loaded_image,
    operation='edges',
    output_dir=output_path,
)

if (image_count > 0
        and backend_ready
        and not queue.empty()):
    start_pipeline()

# ✅ Correcto: string larga
message = (
    'El backend CUDA no está disponible en esta máquina. '
    'Se utilizará el backend CPU como fallback.'
)

# ❌ Incorrecto: backslash para continuar
if image_count > 0 and backend_ready \
        and not queue.empty():
    start_pipeline()
```

---

### 3.3 Paréntesis

Usar paréntesis con moderación. No usar en sentencias `return` o `if` salvo que sea necesario para continuar línea.

```python
# ✅ Correcto
return image_path
return processed_image, elapsed_ms
if not backend_ready:
    raise RuntimeError('Backend not initialized.')

# ❌ Incorrecto
return (image_path)
if (not backend_ready):
    ...
```

---

### 3.4 Indentación

**4 espacios.** Nunca tabs. Las líneas de continuación se alinean con el delimitador de apertura o usan sangría colgante de 4 espacios.

```python
# ✅ Correcto: alineado con delimitador
result = process_image(image, operation,
                       output_dir, quality)

# ✅ Correcto: sangría colgante (nada en la primera línea)
result = process_image(
    image,
    operation,
    output_dir,
    quality,
)

# ❌ Incorrecto: contenido en la primera línea + sangría inconsistente
result = process_image(image, operation,
    output_dir, quality)   # 4 espacios cuando debería alinear
```

**Comas finales:** usar coma al final cuando el cierre `]`, `)` o `}` está en línea separada:

```python
# ✅ Correcto
operations = [
    'grayscale',
    'edges',
    'blur',
    'equalize',
]

# ❌ Incorrecto
operations = [
    'grayscale',
    'edges',
    'blur',
    'equalize',]   # cierre en la misma línea que el último elemento
```

---

### 3.5 Líneas en blanco

- **2 líneas en blanco** entre definiciones de nivel superior (funciones y clases).
- **1 línea en blanco** entre métodos dentro de una clase.
- **1 línea en blanco** entre el docstring de clase y el primer método.
- **Sin línea en blanco** después de una línea `def`.

```python
class CUDABackend(GPUBackend):
    """Backend para GPUs NVIDIA usando Numba CUDA."""

    def __init__(self) -> None:
        self._device = self._init_device()

    def process(self, image: np.ndarray, operation: str) -> np.ndarray:
        ...

    def _init_device(self):
        ...


class OpenCLBackend(GPUBackend):    # 2 líneas en blanco antes
    ...
```

---

### 3.6 Espacios en blanco

```python
# ✅ Sin espacios dentro de paréntesis, corchetes o llaves
result = process(image, {'mode': 'fast'}, [1, 2, 3])

# ❌ Incorrecto
result = process( image, { 'mode': 'fast' }, [ 1, 2, 3 ] )

# ✅ Sin espacio antes de coma, punto y coma, dos puntos
print(x, y)
x, y = y, x

# ❌ Incorrecto
print(x , y)
x , y = y , x

# ✅ Sin espacio antes de paréntesis de argumentos o índice
process(image)
data['key']

# ❌ Incorrecto
process (image)
data ['key']

# ✅ Sin espacios en keyword arguments (sin anotación de tipo)
def process(image, operation='grayscale'):
    backend.run(image=image, op=operation)

# ✅ Con espacio cuando hay anotación de tipo
def process(image: np.ndarray, operation: str = 'grayscale') -> np.ndarray:
    ...

# ❌ Incorrecto
def process(image: np.ndarray, operation: str='grayscale'):   # falta espacio
    backend.run(image = image)                                 # sobra espacio
```

---

### 3.7 Comentarios y docstrings

#### Formato de docstring

Siempre usar triple comilla doble `"""`. La primera línea es un resumen en una sola línea, terminado en punto, signo de pregunta o exclamación.

```python
def get_backend() -> GPUBackend:
    """Detecta el hardware disponible y retorna el backend óptimo.

    Ejecuta la detección en orden de prioridad: CUDA → OpenCL → CPU.
    La selección es transparente para el resto del pipeline.

    Returns:
        Una instancia del mejor backend disponible en el sistema.

    Raises:
        RuntimeError: Si ningún backend puede inicializarse.
    """
    ...
```

#### Docstrings de módulo

```python
"""Pipeline de procesamiento de imágenes con soporte multi-backend.

Este módulo implementa la detección automática de backend GPU y expone
la interfaz unificada GPUBackend para uso en el pipeline productor-consumidor.

Typical usage example:

    backend = get_backend()
    result = backend.process(image, 'grayscale')
"""
```

#### Docstrings de funciones y métodos

Obligatorio cuando la función es parte de la API pública, tiene lógica no obvia, o supera 10 líneas. Secciones estándar:

```python
def process_batch(
    self,
    images: list[np.ndarray],
    operation: str,
    max_concurrent: int = 4,
) -> list[np.ndarray]:
    """Procesa un lote de imágenes aplicando la operación indicada.

    Las imágenes se procesan en paralelo usando el semáforo interno
    para limitar la concurrencia en la GPU y evitar saturación de VRAM.

    Args:
        images: Lista de arrays NumPy con las imágenes a procesar.
            Cada imagen debe tener shape (H, W, C) con dtype uint8.
        operation: Nombre de la transformación a aplicar.
            Valores válidos: 'grayscale', 'edges', 'blur', 'equalize'.
        max_concurrent: Número máximo de imágenes procesadas simultáneamente.
            Reducir si hay errores de memoria de video.

    Returns:
        Lista de arrays procesados en el mismo orden que la entrada.

    Raises:
        ValueError: Si `operation` no es un valor válido.
        MemoryError: Si la imagen supera la VRAM disponible.
    """
    ...
```

#### Docstrings de clases

```python
class MetricsCollector:
    """Agrega y almacena métricas de rendimiento del pipeline de forma thread-safe.

    Attributes:
        total_images: Cantidad total de imágenes procesadas en la sesión.
        total_elapsed_ms: Tiempo acumulado de procesamiento en milisegundos.
    """

    def __init__(self) -> None:
        """Inicializa el colector con contadores en cero."""
        self.total_images: int = 0
        self.total_elapsed_ms: float = 0.0
        self._lock = threading.Lock()
```

#### Comentarios de bloque e inline

```python
# ✅ Correcto: comentar el porqué, no el qué
# Usamos semáforo en lugar de lock porque necesitamos permitir
# múltiples accesos concurrentes acotados, no exclusión mutua.
_gpu_semaphore = threading.Semaphore(MAX_CONCURRENT)

result = image & (image - 1)  # True si image es 0 o potencia de 2

# ❌ Incorrecto: describe lo obvio
# Creamos un semáforo
_gpu_semaphore = threading.Semaphore(MAX_CONCURRENT)
```

Los comentarios inline empiezan al menos 2 espacios después del código, con `#` seguido de un espacio.

---

### 3.8 Strings

Usar f-strings, `%` o `.format()` para interpolar. No concatenar con `+` en loops.

```python
# ✅ Correcto: f-string (preferido en código nuevo)
print(f'Backend activo: {backend_name} — Dispositivo: {device_id}')
logging.info('Procesando imagen %d de %d', current, total)  # logging usa %

# ✅ Correcto: acumular en lista y unir
parts = ['<report>']
for result in results:
    parts.append(f'<image name="{result.name}" ms="{result.elapsed_ms}"/>')
parts.append('</report>')
report = ''.join(parts)

# ❌ Incorrecto: concatenación con + en loop (complejidad cuadrática)
report = '<report>'
for result in results:
    report += f'<image name="{result.name}"/>'
report += '</report>'

# Consistencia de comillas: elegir ' o " y mantenerlo en todo el archivo
# Usar la otra variante para evitar escapes
message = "Backend 'CUDA' no disponible."   # evita backslash
```

**Logging:** siempre pasar el patrón como string literal, nunca f-string:

```python
# ✅ Correcto
logging.info('Imagen procesada: %s en %.2f ms', image_name, elapsed)

# ❌ Incorrecto
logging.info(f'Imagen procesada: {image_name} en {elapsed:.2f} ms')
```

---

### 3.9 Archivos y recursos

Usar `with` para gestionar recursos. Nunca dejar archivos o sockets abiertos.

```python
# ✅ Correcto
with open(output_path, 'wb') as f:
    f.write(processed_image.tobytes())

# ✅ Correcto: múltiples recursos
with (
    open(input_path, 'rb') as src,
    open(output_path, 'wb') as dst,
):
    dst.write(process(src.read()))

# ❌ Incorrecto: confiar en el GC para cerrar
f = open(output_path, 'wb')
f.write(data)   # si falla, el archivo queda abierto
```

---

### 3.10 Comentarios TODO

Formato estándar: `TODO:` seguido de referencia al issue y descripción con guion.

```python
# ✅ Correcto (estilo moderno recomendado)
# TODO: github.com/org/parallelvision/issues/12 - Implementar kernel OpenCL para equalize.
# TODO: github.com/org/parallelvision/issues/8 - Agregar soporte para imágenes RGBA.

# También aceptado (estilo anterior)
# TODO(github.com/org/parallelvision/issues/12): Implementar kernel OpenCL.

# ❌ Evitar: referencia a persona sin issue
# TODO: @tomás - arreglar esto
# TODO: arreglar el semáforo algún día
```

---

### 3.11 Orden de imports

Los imports van al inicio del archivo, después del docstring de módulo y antes de constantes. Se organizan en grupos separados por una línea en blanco:

```python
"""Módulo de detección de backend GPU para ParallelVision."""

# 1. __future__
from __future__ import annotations

# 2. Biblioteca estándar de Python
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Sequence

# 3. Librerías de terceros
import numpy as np

# 4. Imports del propio proyecto
from parallelvision.core.backend import GPUBackend
from parallelvision.core.metrics import MetricsCollector
```

Dentro de cada grupo, ordenar alfabéticamente por path de módulo (ignorando mayúsculas/minúsculas).

---

### 3.12 Nomenclatura (Naming)

#### Tabla de referencia rápida

| Tipo | Público | Interno |
|---|---|---|
| Paquetes | `lower_with_under` | — |
| Módulos | `lower_with_under` | `_lower_with_under` |
| Clases | `CapWords` | `_CapWords` |
| Excepciones | `CapWords` | — |
| Funciones | `lower_with_under()` | `_lower_with_under()` |
| Constantes globales/de clase | `CAPS_WITH_UNDER` | `_CAPS_WITH_UNDER` |
| Variables globales/de clase | `lower_with_under` | `_lower_with_under` |
| Variables de instancia | `lower_with_under` | `_lower_with_under` (protegida) |
| Métodos | `lower_with_under()` | `_lower_with_under()` (protegido) |
| Parámetros de función | `lower_with_under` | — |
| Variables locales | `lower_with_under` | — |

#### Ejemplos aplicados al proyecto

```python
# Módulos
# backend.py, queue_manager.py, metrics_collector.py

# Constantes
MAX_GPU_BATCH_SIZE = 64
VALID_OPERATIONS = ('grayscale', 'edges', 'blur', 'equalize')
_DEFAULT_THREAD_COUNT = 4       # interna al módulo

# Clases
class GPUBackend: ...
class CUDABackend(GPUBackend): ...
class BackendDetectionError(RuntimeError): ...

# Funciones y métodos
def get_backend() -> GPUBackend: ...
def process_batch(images, operation): ...
def _validate_operation(operation: str) -> None: ...   # interna

# Variables
active_backend: GPUBackend
elapsed_ms: float
_lock = threading.Lock()        # interna a la clase/módulo
```

#### Nombres a evitar

```python
# ❌ Nombres de un solo carácter (salvo contadores i, j, k, e, f)
b = get_backend()   # poco descriptivo
r = process(img)    # poco descriptivo

# ✅ Correcto
backend = get_backend()
result = process(image)

# ❌ Incluir el tipo en el nombre
image_list = []           # redundante
backend_dict = {}         # redundante
id_to_name_dict = {}      # redundante

# ✅ Correcto
images = []
backends = {}
names_by_id = {}

# ❌ Abreviaturas no estándar
proc_img()      # procesar imagen
calc_spdup()    # calcular speedup

# ✅ Correcto
process_image()
calculate_speedup()
```

---

### 3.13 Función main

Todo archivo ejecutable debe usar el patrón `if __name__ == '__main__'`:

```python
"""Punto de entrada principal de ParallelVision."""

import sys
from parallelvision.app import ParallelVisionApp


def main() -> None:
    """Inicializa y ejecuta la aplicación ParallelVision."""
    app = ParallelVisionApp()
    app.run()


if __name__ == '__main__':
    main()
```

El código a nivel de módulo se ejecuta al importar, por eso toda la lógica de inicio va dentro de `main()`.

---

### 3.14 Longitud de funciones

Preferir funciones pequeñas y enfocadas. Como guía: **si una función supera las 40 líneas, evaluar si puede dividirse**.

Una función debe hacer una sola cosa. Si tiene más de un nivel de abstracción mezclado, es candidata a ser refactorizada.

```python
# ❌ Función demasiado larga que mezcla niveles de abstracción
def process_all_images(folder_path, operations, output_dir):
    # detectar backend
    # escanear carpeta
    # crear cola
    # lanzar hilos
    # procesar cada imagen
    # guardar resultado
    # calcular métricas
    # generar reporte
    ...   # 80+ líneas

# ✅ Correcto: dividida en funciones con responsabilidad única
def run_pipeline(folder_path: str, operations: list[str], output_dir: str) -> PipelineReport:
    """Ejecuta el pipeline completo de procesamiento."""
    backend = get_backend()
    image_paths = scan_folder(folder_path)
    results = process_images(image_paths, operations, backend)
    save_results(results, output_dir)
    return generate_report(results)
```

---

### 3.15 Anotaciones de tipo — detalles

#### NoneType y opcionales

```python
# ✅ Correcto: Python 3.10+ (union con |)
def process(self, image: np.ndarray | None, operation: str = 'grayscale') -> np.ndarray | None:
    ...

# ✅ Correcto: Python < 3.10 (Optional)
from typing import Optional
def process(self, image: Optional[np.ndarray], operation: str = 'grayscale') -> Optional[np.ndarray]:
    ...

# ❌ Incorrecto: None implícito
def process(self, image: np.ndarray = None) -> np.ndarray:   # no documenta que puede ser None
    ...
```

#### Tipos abstractos preferidos

```python
from collections.abc import Sequence, Mapping

# ✅ Correcto: tipo abstracto (más flexible)
def process_batch(images: Sequence[np.ndarray]) -> list[np.ndarray]:
    ...

# ❌ Innecesariamente concreto
def process_batch(images: list[np.ndarray]) -> list[np.ndarray]:
    ...
```

#### Alias de tipos complejos

```python
from typing import TypeAlias

# Alias para tipos complejos usados frecuentemente
ImageArray: TypeAlias = np.ndarray
MetricsRecord: TypeAlias = dict[str, float | str | int]
ProcessingResult: TypeAlias = tuple[ImageArray, MetricsRecord]
```

#### Ruptura de línea con anotaciones

```python
# ✅ Correcto: un parámetro por línea cuando la firma es larga
def process_batch(
    self,
    images: list[np.ndarray],
    operation: str,
    output_dir: str | None = None,
    max_concurrent: int = 4,
) -> list[np.ndarray]:
    ...
```

---

## 4. Palabra final

> **SÉ CONSISTENTE.**

Si estás editando código existente, tomá unos minutos para observar el estilo del código que te rodea y adoptarlo. El objetivo de una guía de estilo es tener un vocabulario común de codificación para que el equipo pueda concentrarse en **qué** se está haciendo, no en **cómo** está escrito.

La consistencia local es igual de importante que las reglas globales. Si el código que agregás se ve radicalmente diferente del código que lo rodea, dificulta la lectura del archivo completo.

---

*Basado en [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) — adaptado para ParallelVision, UNLAM 2026*
