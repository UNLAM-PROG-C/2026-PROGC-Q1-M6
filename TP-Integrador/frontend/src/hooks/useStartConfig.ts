import { useMemo, useState } from "react";
import type { StartConfig } from "@/api/types";

const DEFAULT_WORKERS = 4;

export interface StartConfigForm {
  inputDir: string;
  outputDir: string;
  operations: Set<string>;
  workers: number;
  ready: boolean;
  setInputDir: (path: string) => void;
  setOutputDir: (path: string) => void;
  toggleOperation: (value: string) => void;
  setWorkers: (count: number) => void;
  toPayload: () => StartConfig;
}

/** Estado del formulario de configuración del procesamiento (#20). */
export function useStartConfig(): StartConfigForm {
  const [inputDir, setInputDir] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [operations, setOperations] = useState<Set<string>>(new Set());
  const [workers, setWorkers] = useState(DEFAULT_WORKERS);

  const toggleOperation = (value: string) =>
    setOperations((prev) => {
      const next = new Set(prev);
      if (next.has(value)) {
        next.delete(value);
      } else {
        next.add(value);
      }
      return next;
    });

  const ready = useMemo(
    () => Boolean(inputDir && outputDir && operations.size > 0),
    [inputDir, outputDir, operations],
  );

  const toPayload = (): StartConfig => ({
    input_dir: inputDir,
    output_dir: outputDir,
    operations: [...operations],
    workers,
  });

  return {
    inputDir,
    outputDir,
    operations,
    workers,
    ready,
    setInputDir,
    setOutputDir,
    toggleOperation,
    setWorkers,
    toPayload,
  };
}
