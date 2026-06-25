# ParallelVision · Dashboard (frontend)

SPA del **Dashboard en tiempo real** (Capa 5) de ParallelVision. Consume la
API FastAPI de `api/` y recibe el progreso del pipeline por WebSocket.

## Stack

- **React 18 + TypeScript + Vite**
- **TailwindCSS** + componentes estilo **shadcn/ui** (Radix UI)
- **ESLint** (calidad de código TS/TSX)

## Requisitos

- Node.js 18+ y npm.
- El backend FastAPI corriendo en `http://localhost:8000`
  (ver raíz del proyecto: `uvicorn api.main:app --reload`).

## Cómo correrlo

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Vite hace de **proxy** de `/api` y `/ws` hacia `http://localhost:8000`
(`vite.config.ts`), así que en desarrollo no hace falta configurar CORS.

## Scripts

| Comando         | Descripción                              |
| --------------- | ---------------------------------------- |
| `npm run dev`   | Servidor de desarrollo con HMR.          |
| `npm run build` | Type-check (`tsc -b`) + build de Vite.   |
| `npm run lint`  | ESLint sobre `src/`.                     |

## Estructura

```
src/
├── api/        # client.ts (fetch) + types.ts (DTOs)
├── hooks/      # useBackend, useOperations, useProgress, useStartConfig
├── components/ # BackendIndicator, ConfigPanel, ProgressPanel, FolderPicker…
│   └── ui/     # primitivas estilo shadcn (button, card, progress, dialog…)
├── App.tsx     # layout header + dos columnas
└── main.tsx
```

## Alcance (Fase 2)

Esqueleto del dashboard (#20) + barra de progreso global por WebSocket (#21).
El botón **Iniciar procesamiento** valida y envía la configuración a
`POST /api/start`, que en esta fase es un *stub*. La ejecución real del
pipeline y las métricas en vivo (velocidad / tiempo / ETA) llegan en **Fase 3
(#28)**.
