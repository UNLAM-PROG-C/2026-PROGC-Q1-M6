export interface BackendInfo {
  backend_name: string;
  device_info: string;
  is_gpu: boolean;
}

export interface Operation {
  value: string;
  label: string;
}

export interface DirEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface BrowseResult {
  path: string;
  parent: string | null;
  entries: DirEntry[];
}

export interface StartConfig {
  input_dir: string;
  output_dir: string;
  operations: string[];
  workers: number;
}

export interface StartResponse {
  accepted: boolean;
  message: string;
}

export interface ProgressUpdate {
  current: number;
  total: number;
  percent: number;
  running: boolean;
}
