import { useMemo, useState } from "react";
import type { StartConfig } from "@/api/types";

const DEFAULT_WORKERS = 4;

export interface StartConfigForm {
  inputDir: string;
  outputDir: string;
  operation: string;
  workers: number;
  ready: boolean;
  setInputDir: (path: string) => void;
  setOutputDir: (path: string) => void;
  setOperation: (value: string) => void;
  setWorkers: (count: number) => void;
  toPayload: () => StartConfig;
}

/** Estado del formulario de configuración del procesamiento (#20). */
export function useStartConfig(): StartConfigForm {
  const [inputDir, setInputDir] = useState("sample_images");
  const [outputDir, setOutputDir] = useState("result");
  const [operation, setOperation] = useState<string>("grayscale");
  const [workers, setWorkers] = useState(DEFAULT_WORKERS);

  const ready = useMemo(
    () => Boolean(inputDir && outputDir && operation),
    [inputDir, outputDir, operation],
  );

  const toPayload = (): StartConfig => ({
    input_dir: inputDir,
    output_dir: outputDir,
    operation: operation,
    workers,
  });

  return {
    inputDir,
    outputDir,
    operation,
    workers,
    ready,
    setInputDir,
    setOutputDir,
    setOperation,
    setWorkers,
    toPayload,
  };
}
