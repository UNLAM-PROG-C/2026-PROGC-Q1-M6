# ParallelVision 👁️🚀

## Descripción del Proyecto

**ParallelVision** es un pipeline de procesamiento masivo de imágenes diseñado para aplicar diversas operaciones y filtros a grandes volúmenes de archivos de forma eficiente. 

El problema principal que resuelve es el cuello de botella que se genera al procesar miles de imágenes de manera secuencial. Para solucionarlo, ParallelVision aprovecha al máximo los recursos de hardware disponibles, aplicando técnicas de concurrencia y utilizando tanto procesamiento asíncrono en la CPU como aceleración por GPU.

### Tabla de compatibilidad de Hardware

El sistema tiene detección automática de hardware, utilizando el mejor disponible según la siguiente prioridad:

| Hardware | Backend | Librería | Prioridad |
|---|---|---|---|
| NVIDIA GPU | CUDA | Numba | 1 (Más Alta) |
| AMD / Intel GPU | OpenCL | PyOpenCL | 2 |
| Cualquier CPU | Multithreading | concurrent.futures | 3 (Más Baja) |

---

## Arquitectura 🏗️

El sistema utiliza un diseño estructurado bajo patrones de concurrencia organizados en las siguientes 5 capas:

```text
[ Capa 1: Carga Lectura] -> Escanea directorios y encola paths rápidamente
        |
        v
[ Capa 2: Cola Entrada ] -> Estructura thread-safe (Productor-Consumidor)
        |
        v
[ Capa 3: Workers ] ------> Pool de hilos CPU + Kernels GPU
        |      \
        |       \--------> [ Cola de E/S ] -> [ Hilo de Guardado en Disco ]
        v
[ Capa 4: Agregador ] ----> Recolector de métricas y cálculo de Speedup
        |
        v
[ Capa 5: Dashboard ] ----> Panel de control visual interactivo
```

### Patrón Strategy para detección de Backend

Para lograr la máxima flexibilidad de hardware, el backend implementa el **patrón de diseño Strategy**. Al inicializar el sistema, se detecta el hardware disponible y se instancia dinámicamente el backend correspondiente (`CUDABackend`, `OpenCLBackend` o `CPUBackend`). Todos ellos exponen y respetan la misma interfaz base `GPUBackend` con el método de entrada `process(image, operation)`. De esta forma, el resto del código del pipeline (workers y colas) funciona de manera agnóstica al hardware subyacente.

---

## Dependencias y Requisitos 📋

- **Python 3.11** o superior.
- **Requisitos opcionales para GPU**: 
  - **NVIDIA (CUDA)**: Requiere tener los drivers de NVIDIA actualizados y el CUDA Toolkit instalado.
  - **AMD / Intel (OpenCL)**: Requiere drivers de OpenCL compatibles.

---

## Instalación ⚙️

Sigue estos pasos para clonar el repositorio, crear un entorno virtual e instalar las dependencias:

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd TP-Integrador

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar el entorno virtual
# En Linux / macOS:
source venv/bin/activate
# En Windows (PowerShell):
.\venv\Scripts\activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar script principal
python main.py --help
```

---

## Manual de Usuario 📖

### Uso por Línea de Comandos (CLI)

Es posible prescindir de la interfaz visual y usar el CLI base mediante el script `main.py`:

```bash
python main.py --input-dir ruta/a/imagenes --operation blur
```

#### Argumentos disponibles (CLI)

- `--input-dir`: **(Requerido)** Directorio de imágenes de entrada.
- `--operation`: **(Requerido)** Operación a aplicar (`grayscale`, `edges`, `blur`, `equalize`).
- `--workers`: Cantidad de hilos de trabajo (por defecto utiliza un máximo basado en tu procesador).
- `--queue-size`: Tamaño máximo de la cola en memoria (por defecto 100).
- `--report`: Ruta para generar el reporte de métricas CSV (ej. `resultado.csv`).
- `--output-dir`: Directorio donde se guardarán las imágenes procesadas (por defecto `result/`).
- `--no-save`: Bandera para procesar imágenes en memoria sin guardarlas en disco (útil para benchmarking de puro rendimiento CPU/GPU).

#### Cómo exportar el reporte CSV
Para generar un reporte analítico del rendimiento de las operaciones, añade el parámetro `--report` seguido de un nombre de archivo:

```bash
python main.py --input-dir public/images/in --operation blur --report metricas_blur.csv
```
El reporte CSV listará todas las métricas procesadas individualmente y puede ser utilizado para documentar en hojas de cálculo el desempeño exacto del equipo bajo pruebas de carga.

### Uso del Dashboard Interactivo (GUI)

La interfaz gráfica y el dashboard funcionan mediante una arquitectura cliente-servidor (Backend FastAPI + Frontend React/Vite).

#### 1. Preparar imágenes de prueba
Puedes descargar imágenes aleatorias para probar el sistema usando el script de descarga incorporado:

```bash
python scripts/descargar_imagenes.py -n 100 -d public/images/in
```
*(Usa `-n` para indicar la cantidad y `-d` para el directorio destino).*

#### 2. Levantar el Servidor Backend (API)
El backend expone la API REST y los WebSockets para la comunicación en tiempo real. Ejecuta el servidor desde la raíz del proyecto:

```bash
python -m uvicorn api.main:app --reload
```
*(El backend correrá en `http://localhost:8000`).*

#### 3. Levantar el Frontend (Dashboard)
El frontend proporciona el panel interactivo. En una nueva terminal, entra a la carpeta, instala las dependencias y corre el servidor de desarrollo:

```bash
cd frontend
npm install
npm run dev
```
*(El frontend correrá en la url que indique Vite, ej. `http://localhost:5173`).*

#### 4. Paso a Paso en la Interfaz

1. **Seleccionar Carpetas**: En el panel de la UI, utiliza los campos correspondientes para indicar el directorio de entrada (donde están tus imágenes originales) y el directorio de salida (donde se guardarán las modificadas).
2. **Elegir Operaciones**: Selecciona la transformación o filtro deseado (`grayscale`, `edges`, `blur`, `equalize`) desde el menú de opciones.
3. **Iniciar**: Haz clic en el botón de iniciar procesamiento. Podrás observar en tiempo real cómo avanza la barra de progreso para las distintas tareas.

#### 5. Indicador de Backend y Gráfico de Speedup
- **Explicación del indicador de backend**: En el dashboard podrás visualizar si tu procesamiento está derivándose a `CUDA`, `OpenCL` o utilizando el `CPU`. Esto confirma cuál estrategia de hardware se auto-seleccionó al arrancar el servidor según tus componentes.
- **Cómo interpretar el gráfico de speedup**: El sistema genera un gráfico que compara la eficiencia real (con la aceleración GPU) frente a un escenario simulado puro de CPU. Un valor de *Speedup de 3.5x* indica que el procesamiento terminó 3.5 veces más rápido de lo que habría tardado si no se hubiese usado aceleración por hardware.