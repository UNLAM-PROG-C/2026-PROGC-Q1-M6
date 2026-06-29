import { useCallback, useEffect, useRef, useState } from "react";
import { getChart } from "@/api/client";
import type { ChartPoint } from "@/api/types";

const POLL_INTERVAL_MS = 2000;

/** Polling a /api/metrics/chart mientras running; fetch final al terminar. */
export function useSpeedupChart(running: boolean, hasData: boolean) {
  const [data, setData] = useState<ChartPoint[]>([]);
  const timer = useRef<number | undefined>(undefined);

  const fetchChart = useCallback(async () => {
    try {
      setData(await getChart());
    } catch {
      // ignore errors de red durante el polling
    }
  }, []);

  useEffect(() => {
    if (!running && !hasData) return;
    fetchChart();
    if (running) {
      timer.current = window.setInterval(fetchChart, POLL_INTERVAL_MS);
    }
    return () => window.clearInterval(timer.current);
  }, [running, hasData, fetchChart]);

  return data;
}
