# Probar ParallelVision con GPU CUDA en Google Colab

Google Colab ofrece una GPU **NVIDIA T4** con drivers CUDA preinstalados. Esta guía corre el **backend FastAPI en Colab**, lo expone con un túnel **ngrok**, y deja el **frontend React corriendo local** apuntado a esa URL.

---

## Paso 1 — Colab con runtime GPU

1. Abrí [`colab/ParallelVision_GPU.ipynb`](../colab/ParallelVision_GPU.ipynb) en Colab.
2. **Runtime → Cambiar tipo de entorno de ejecución → GPU** (T4). **Realizar antes**
   de ejecutar cualquier celda.
3. Ejecutá las celdas en orden:

```python
# Celda 1 — verificar GPU
!nvidia-smi
from numba import cuda
print(cuda.gpus)            # debe listar la Tesla T4
```

```python
# Celda 2 — traer el repo
!git clone -b develop https://github.com/UNLAM-PROG-C/2026-PROGC-Q1-M6.git
%cd 2026-PROGC-Q1-M6/TP-Integrador
```

```python
# Celda 3 — deps
!pip install -r requirements.txt
```

```python
# Celda 4 — (opcional) smoke test del backend CUDA
!python -m pytest tests/test_cuda_backend.py -v
```

```python
# Celda 5 — túnel ngrok + servidor
from pyngrok import ngrok
ngrok.set_auth_token("TU_AUTHTOKEN")   # https://dashboard.ngrok.com/get-started/your-authtoken
public_url = ngrok.connect(8000)
print("Pegá esto en frontend/.env.local -> VITE_API_TARGET =", public_url)
!uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Notas importantes:

- `--host 0.0.0.0` es **obligatorio**: el default `127.0.0.1` no es alcanzable
  por el túnel.
- ngrok free da una **URL aleatoria por sesión**. Para fijarla, reservá tu
  **dominio estático gratuito** en el dashboard de ngrok y arrancá el túnel con
  `ngrok http --domain=<tu-dominio>.ngrok-free.app 8000`. Así `.env.local` se
  configura **una sola vez**.
- La celda de uvicorn **queda corriendo** (bloqueante). Es lo esperado: dejala
  viva mientras usás el dashboard.

---

## Paso 2 — Frontend local apuntado a Colab

```bash
cd TP-Integrador/frontend
cp .env.example .env.local        # solo la primera vez
```

Editá `frontend/.env.local` con la URL que imprimió ngrok:

```env
VITE_API_TARGET=https://xxxx.ngrok-free.app
```

Luego:

```bash
npm install                       # solo la primera vez
npm run dev                       # reiniciá si ya estaba corriendo
```

Abrí `http://localhost:5173`. `.env.local` está gitignoreado, así que la URL
efímera no se commitea.

---

## Verificación end-to-end

1. **Colab:** `cuda.gpus` lista la T4 → `pytest tests/test_cuda_backend.py` pasa →
   uvicorn arriba → ngrok imprime la URL pública.
2. **Local:** URL en `frontend/.env.local` → `npm run dev` → `localhost:5173`.
3. **Backend activo = CUDA:** el dashboard (endpoint `/api/backend`) muestra
   **CUDA**, no CPU ni OpenCL.
4. **Pipeline real:** lanzá un procesamiento desde la UI sobre `public/images/in/` →
   el WebSocket `/ws/progress` actualiza la barra en tiempo real → la galería carga.
5. **Métricas:** `/api/metrics/summary` muestra speedup GPU vs CPU > 1.

## Troubleshooting

- **Página interstitial de ngrok:** ya está mitigada con el header
  `ngrok-skip-browser-warning` que agrega el proxy de Vite. Si aún aparece, refrescá.
- **El WS no conecta:** confirmá que la celda de uvicorn sigue viva y que la URL en
  `.env.local` es `https://` (Vite la upgradea a `wss://` solo).
- **`cuda.gpus` vacío o error de driver:** el runtime no es GPU. Cambialo en
  *Runtime → Cambiar tipo de entorno* y reejecutá desde la Celda 1.

- **Error 500 en el front al lanzar un procesamiento (`/api/start`):** este 500
  **no lo genera FastAPI**. El backend nunca devuelve 500 desde el pipeline: el
  `/api/start` responde `202` de inmediato y cualquier error de procesamiento se
  atrapa y solo se loguea (`api/routes.py`). El 500 lo devuelve **el proxy de
  Vite** cuando no logra hablar con el upstream. Se reconoce por este log en la
  terminal del front:

  ```text
  [vite] http proxy error: /api/start
  Error: Client network socket disconnected before secure TLS connection was established
  ```

  **Diagnóstico:** si el error es real, en Colab **no aparece** ningún
  `Solicitud de inicio: ... op=<...>` para ese intento. Eso confirma que el
  request **murió en el salto Vite → ngrok** y nunca atravesó el túnel; no es un
  bug de la operación (p. ej. `equalize`), que está soportada en los tres
  backends.

  **Causa:** inestabilidad del túnel **ngrok-free** — la conexión TLS hacia el
  edge de ngrok se resetea/cae antes de completar el handshake (límites de
  conexiones del plan gratuito, cortes del túnel, reconexiones en frío). Es
  **intermitente** y suele pegar tras un rato de inactividad.

  **Workarounds cuando vuelve a pasar:**
  1. **Reintentar** la acción — al ser intermitente, suele pasar al segundo
     intento.
  2. **Regenerar el túnel:** reejecutar la celda de ngrok/uvicorn en Colab y
     actualizar `VITE_API_TARGET` en `.env.local` (o usar un **dominio
     reservado** para no tocar el `.env`).
  3. Para una demo crítica, considerar **ngrok pago** (túnel estable) o correr el
     **backend local** (sin GPU) para descartar la red.
