import { useEffect, useRef, useState } from "react";
import { getSummary } from "@/api/client";
import type { OperationResult } from "@/api/types";

/** Fetch a /api/metrics/summary cuando el run pasa de true → false. */
export function useResultsSummary(running: boolean, current: number) {
  const [summary, setSummary] = useState<OperationResult[]>([]);
  const prevRunning = useRef(running);

  useEffect(() => {
    const wasRunning = prevRunning.current;
    prevRunning.current = running;
    if (wasRunning && !running && current > 0) {
      getSummary().then(setSummary).catch(() => {});
    }
  }, [running, current]);

  return summary;
}
