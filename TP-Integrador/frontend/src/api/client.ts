import type {
  BackendInfo,
  BrowseResult,
  ChartPoint,
  Operation,
  OperationResult,
  StartConfig,
  StartResponse,
} from "@/api/types";

const API_BASE = "/api";

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Error ${res.status} al llamar a ${url}`);
  }
  return res.json() as Promise<T>;
}

export function getBackend(): Promise<BackendInfo> {
  return getJson<BackendInfo>(`${API_BASE}/backend`);
}

export function getOperations(): Promise<Operation[]> {
  return getJson<Operation[]>(`${API_BASE}/operations`);
}

export function browse(path = ""): Promise<BrowseResult> {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return getJson<BrowseResult>(`${API_BASE}/browse${query}`);
}

export function getChart(): Promise<ChartPoint[]> {
  return getJson<ChartPoint[]>(`${API_BASE}/metrics/chart`);
}

export function getSummary(): Promise<OperationResult[]> {
  return getJson<OperationResult[]>(`${API_BASE}/metrics/summary`);
}

export function downloadMetricsCsv(): void {
  const a = document.createElement("a");
  a.href = `${API_BASE}/metrics/export`;
  a.download = "parallelvision_report.csv";
  a.click();
}

export async function start(config: StartConfig): Promise<StartResponse> {
  const res = await fetch(`${API_BASE}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body?.detail ?? `Error ${res.status} al iniciar el procesamiento`;
    throw new Error(detail);
  }
  return res.json() as Promise<StartResponse>;
}
