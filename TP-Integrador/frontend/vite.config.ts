import path from "node:path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_TARGET || "http://localhost:8000";
  const proxyOpts = {
    target: apiTarget,
    changeOrigin: true,
    headers: { "ngrok-skip-browser-warning": "true" },
  };
  return {
    plugins: [react()],
    resolve: {
      alias: { "@": path.resolve(__dirname, "./src") },
    },
    server: {
      proxy: {
        "/api": proxyOpts,
        "/ws": { ...proxyOpts, ws: true },
      },
    },
  };
});
