# Estrategia de Branching — ParallelVision

> **Programación Concurrente · UNLAM 2026**  
> Este documento define la estrategia de control de versiones, convenciones de nombres y flujo de trabajo Git adoptados por el equipo para el desarrollo de ParallelVision.

---

## Índice

1. [Modelo adoptado](#1-modelo-adoptado)
2. [Estructura de ramas](#2-estructura-de-ramas)
3. [Ramas por integrante](#3-ramas-por-integrante)
4. [Convenciones de nomenclatura](#4-convenciones-de-nomenclatura)
5. [Convenciones de commits](#5-convenciones-de-commits)
6. [Flujo de trabajo diario](#6-flujo-de-trabajo-diario)
7. [Pull Requests y revisión de código](#7-pull-requests-y-revisión-de-código)
8. [Commit inicial — contratos de interfaz](#8-commit-inicial--contratos-de-interfaz)
9. [Gestión de Issues](#9-gestión-de-issues)
10. [Diagrama general del flujo](#10-diagrama-general-del-flujo)
11. [Resolución de conflictos](#11-resolución-de-conflictos)
12. [Checklist de entrega](#12-checklist-de-entrega)

---

## 1. Modelo adoptado

El equipo utiliza **GitHub Flow**, una estrategia liviana y lineal especialmente adecuada para proyectos con ciclos de entrega cortos y equipos pequeños.

### ¿Por qué GitHub Flow y no Git Flow?

| Criterio | Git Flow | GitHub Flow ✅ |
|---|---|---|
| Complejidad | Alta (ramas `develop`, `release`, `hotfix`) | Baja (`main` + feature branches) |
| Adecuado para proyectos cortos | No | Sí |
| Curva de aprendizaje | Alta | Baja |
| Integración continua | Compleja | Simple |
| Tamaño de equipo ideal | 5+ personas | 2–6 personas |

Git Flow introduce una sobrecarga de gestión innecesaria para un proyecto de 8 semanas con 4 integrantes. GitHub Flow permite mantener el foco en el desarrollo sin fricción administrativa.

### Principio fundamental

> **`main` siempre contiene código funcional.** Nadie hace push directo a `main`. Todo cambio entra exclusivamente a través de un Pull Request revisado y aprobado.

---

## 2. Estructura de ramas

```
main
├── feature/image-loader-queue
├── feature/cpu-pipeline-backend
├── feature/gpu-cuda-opencl
└── feature/metrics-report
```

### Descripción de cada rama

| Rama | Propósito |
|---|---|
| `main` | Código estable y funcional. Base de la entrega final. |
| `feature/*` | Desarrollo de funcionalidades individuales. Una rama por módulo/integrante. |
| `fix/*` | Corrección de bugs detectados en `main` post-merge. |
| `docs/*` | Cambios exclusivos de documentación (README, este archivo, informe). |

---

## 3. Convenciones de nomenclatura

### Ramas

El nombre de una rama debe ser descriptivo, en minúsculas y usar guiones medios (`-`) como separador.

```
<tipo>/<descripcion-corta>
```

**Tipos válidos:**

| Tipo | Uso |
|---|---|
| `feature` | Nueva funcionalidad |
| `fix` | Corrección de un bug |
| `docs` | Documentación únicamente |
| `test` | Pruebas o experimentos que no van a `main` |
| `refactor` | Mejora interna sin cambio de comportamiento |

**Ejemplos correctos:**

```bash
feature/image-loader-queue
feature/gpu-cuda-opencl
fix/race-condition-lock
fix/queue-deadlock
docs/readme-manual-usuario
refactor/backend-strategy-pattern
```

**Ejemplos incorrectos:**

```bash
MiRama              # ❌ sin tipo, sin guiones
feature/CUDA        # ❌ mayúsculas
arreglo-bug         # ❌ sin tipo
feature/cosas       # ❌ no descriptivo
```

---

## 5. Convenciones de commits

Se utiliza el estándar **Conventional Commits** adaptado al español, con el siguiente formato:

```
<tipo>: <descripción en imperativo, minúsculas, sin punto final>
```

### Tipos de commit

| Tipo | Cuándo usarlo |
|---|---|
| `feat` | Se agrega una funcionalidad nueva |
| `fix` | Se corrige un bug |
| `refactor` | Cambio interno sin alterar comportamiento externo |
| `test` | Se agregan o modifican pruebas |
| `docs` | Cambios en documentación |
| `chore` | Tareas de mantenimiento (dependencias, configs) |
| `perf` | Mejora de rendimiento |

### Ejemplos de commits correctos

```bash
# Funcionalidades nuevas
feat: implementar detección automática de backend CUDA
feat: agregar semáforo para limitar lotes GPU simultáneos
feat: crear clase base GPUBackend con interfaz process()

# Correcciones
fix: corregir condición de carrera en agregador de resultados
fix: resolver deadlock en cola cuando el pipeline se vacía

# Documentación
docs: agregar manual de usuario al README
docs: completar sección de conclusiones en el informe

# Refactors
refactor: separar lógica de detección de backend en módulo propio
refactor: extraer constantes de configuración de hilos

# Mantenimiento
chore: agregar dependencias numba y pyopencl al requirements.txt
chore: configurar .gitignore para excluir __pycache__ y .env
```

### Reglas adicionales de commit

- **Commits pequeños y frecuentes**: un commit por cambio lógico, no acumular cambios de varios días en un solo commit.
- **Nunca commitear archivos innecesarios**: imágenes de prueba grandes, entornos virtuales (`venv/`), archivos `.pyc` o credenciales.
- **El mensaje debe entenderse sin ver el diff**: si alguien lee solo el mensaje, debe saber qué cambió.

---

## 6. Flujo de trabajo diario

El ciclo de trabajo estándar para cada integrante es el siguiente:

### Paso a paso

```bash
# 1. Asegurarse de tener main actualizado
git checkout main
git pull origin main

# 2. Actualizar la rama de trabajo con los últimos cambios de main
git checkout feature/mi-rama
git merge main
# (resolver conflictos si los hay, ver sección 11)

# 3. Trabajar normalmente: editar, probar, commitear
git add archivo_modificado.py
git commit -m "feat: implementar kernel OpenCL para escala de grises"

# 4. Subir cambios al repositorio remoto
git push origin feature/mi-rama

# 5. Cuando el módulo está listo: abrir Pull Request en GitHub
```

### Sincronización con `main`

Cada integrante debe hacer `merge` desde `main` hacia su rama **al menos una vez por semana**, preferentemente al inicio de cada sesión de trabajo. Esto reduce la acumulación de conflictos.

---

## 7. Pull Requests y revisión de código

### Cuándo abrir un PR

Un Pull Request se abre cuando:

- Un módulo o funcionalidad está completo y funcionando.
- Se necesita revisión de un bloque de código importante antes de continuar.
- Se corrigió un bug que afecta a otros módulos.

### Proceso de revisión

1. El autor abre el PR desde su rama hacia `main` en GitHub.
2. Escribe una descripción clara con: **qué se implementó**, **cómo probarlo** y si **cierra algún Issue** (`Closes #N`).
3. **Al menos un compañero** debe revisar el PR antes del merge.
4. El revisor puede: aprobar ✅, solicitar cambios 🔄 o dejar comentarios 💬.
5. Una vez aprobado, el autor hace el merge.
6. La rama de feature se **elimina** luego del merge para mantener el repositorio limpio.

### Plantilla de descripción de PR

```markdown
## ¿Qué se implementó?
Breve descripción del cambio.

## ¿Cómo probarlo?
Pasos para ejecutar o verificar la funcionalidad.

## Issues relacionados
Closes #N

## Checklist
- [ ] El código no rompe funcionalidades existentes
- [ ] Se probó localmente
- [ ] Los nombres de funciones y variables son claros
- [ ] No hay código comentado innecesario
```

---

## 8. Commit inicial — contratos de interfaz

Para evitar que los integrantes se bloqueen entre sí esperando que otro termine su módulo, **en la Semana 1 se hace un commit a `main` con los esqueletos de las clases compartidas**.

Este commit define los **contratos de interfaz**: clases con sus firmas de métodos pero sin implementación real. Así cada integrante puede trabajar contra la interfaz sin necesitar la implementación del otro.

```python
# parallelvision/core/backend.py — commit inicial a main
class GPUBackend:
    """Interfaz común para todos los backends de procesamiento."""

    def process(self, image, operation: str):
        """
        Procesa una imagen aplicando la operación indicada.

        Args:
            image: np.ndarray — imagen de entrada
            operation: str — 'grayscale' | 'edges' | 'blur' | 'equalize'

        Returns:
            np.ndarray — imagen procesada
        """
        raise NotImplementedError


class CUDABackend(GPUBackend):
    """Backend para GPUs NVIDIA. Implementado en feature/gpu-cuda-opencl."""
    def process(self, image, operation: str):
        raise NotImplementedError


class OpenCLBackend(GPUBackend):
    """Backend para GPUs AMD e Intel. Implementado en feature/gpu-cuda-opencl."""
    def process(self, image, operation: str):
        raise NotImplementedError


class CPUBackend(GPUBackend):
    """Backend fallback sin GPU. Implementado en feature/cpu-pipeline-backend."""
    def process(self, image, operation: str):
        raise NotImplementedError


def get_backend() -> GPUBackend:
    """
    Detecta el hardware disponible y retorna el backend óptimo.
    Implementado en feature/cpu-pipeline-backend.
    """
    raise NotImplementedError
```

```python
# parallelvision/core/queue_manager.py — commit inicial a main
import queue

class ImageQueue:
    """Cola thread-safe para el pipeline productor-consumidor."""

    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)

    def put(self, item):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def task_done(self):
        raise NotImplementedError
```

```python
# parallelvision/core/metrics.py — commit inicial a main
import threading

class MetricsCollector:
    """Agregador de resultados con acceso thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = []

    def record(self, image_name: str, backend: str, elapsed: float):
        raise NotImplementedError

    def get_summary(self) -> dict:
        raise NotImplementedError
```

> ⚠️ **Regla:** ningún integrante modifica estos archivos en su propia rama sin coordinarlo con el equipo, ya que todos dependen de ellos.

---

## 9. Gestión de Issues

Se utiliza el sistema de **Issues de GitHub** para trackear tareas, bugs y mejoras.

### Etiquetas recomendadas

| Etiqueta | Color | Uso |
|---|---|---|
| `feature` | Azul | Nueva funcionalidad planificada |
| `bug` | Rojo | Error encontrado |
| `docs` | Gris | Documentación pendiente |
| `blocked` | Naranja | La tarea espera a otra |
| `in progress` | Verde | En desarrollo activo |

### Vincular commits e Issues

Al hacer un commit o PR que resuelve un Issue, incluir en el mensaje:

```bash
# En el commit
fix: corregir deadlock en cola cuando el pipeline se vacía

Closes #7

# En la descripción del PR
Closes #7
Closes #12
```

Esto cierra el Issue automáticamente al hacer merge a `main`.

---

## 10. Diagrama general del flujo

```
main ──●────────────────────────────────────────────────────●── entrega final
       │                                                     │
       │ (commit inicial: interfaces vacías)                 │
       │                                                     │
       ├──[feature/image-loader-queue]──────●──PR✓──merge───┤
       │                                                     │
       ├──[feature/cpu-pipeline-backend]────────────●──PR✓──┤
       │                                                     │
       ├──[feature/gpu-cuda-opencl]──────────────────●──PR✓─┤
       │                                                     │
       └──[feature/metrics-report]────────────────●──PR✓────┘

       ──── desarrollo ────────────────────────────── semana 7-8 ──▶
```

---

## 11. Resolución de conflictos

Los conflictos ocurren cuando dos ramas modificaron la misma sección de un archivo. El proceso para resolverlos es:

```bash
# 1. Actualizar main
git checkout main
git pull origin main

# 2. Mergear main en tu rama
git checkout feature/mi-rama
git merge main

# 3. Git indicará los archivos en conflicto:
#    CONFLICT (content): Merge conflict in parallelvision/core/backend.py

# 4. Abrir el archivo y resolver manualmente:
#    <<<<<<< HEAD          ← tu versión
#    código de tu rama
#    =======
#    código de main
#    >>>>>>> main          ← versión de main

# 5. Luego del fix, marcar como resuelto y commitear
git add parallelvision/core/backend.py
git commit -m "fix: resolver conflicto de merge en backend.py"
```

### Prevención de conflictos

- Hacer merge desde `main` frecuentemente (mínimo una vez por semana).
- Comunicar al equipo antes de modificar archivos compartidos (los del commit inicial).
- Mantener cada módulo bien delimitado: respetar la división de responsabilidades del punto 3.

---

## 12. Checklist de entrega

Antes de la fecha de entrega (01/07–08/07/2026), verificar:

- [ ] `main` contiene el código completo e integrado de todos los módulos.
- [ ] No quedan ramas de feature abiertas sin mergear (o están documentadas como WIP).
- [ ] El historial de commits en `main` es limpio y descriptivo.
- [ ] El `README.md` está actualizado con instrucciones de instalación y uso.
- [ ] El repositorio no contiene archivos innecesarios (imágenes de prueba grandes, `venv/`, `.env`, `__pycache__/`).
- [ ] El `.gitignore` cubre correctamente los archivos a excluir.
- [ ] El cuaderno de Google Colab está subido al repositorio.
- [ ] El informe PDF está adjunto en la plataforma Miel con el nombre correcto (`TP_Integrador_NumerodelGrupo.pdf`).

