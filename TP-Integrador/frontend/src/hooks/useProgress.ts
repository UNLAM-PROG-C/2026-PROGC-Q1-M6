import { useEffect, useRef, useState } from "react";
import type { ProgressUpdate } from "@/api/types";

const RECONNECT_MS = 1500;
const INITIAL: ProgressState = {
  current: 0,
  total: 0,
  percent: 0,
  running: false,
  connected: false,
  speed: 0,
  elapsed: 0,
  eta: 0,
};

export interface ProgressState extends ProgressUpdate {
  connected: boolean;
}

function buildWsUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/progress`;
}

/** Suscribe al WebSocket de progreso y reconecta ante caídas (#21). */
export function useProgress(): ProgressState {
  const [state, setState] = useState<ProgressState>(INITIAL);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;

    const connect = () => {
      socket = new WebSocket(buildWsUrl());
      socket.onopen = () =>
        setState((prev) => ({ ...prev, connected: true }));
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as ProgressUpdate;
        setState({ ...data, connected: true });
      };
      socket.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }));
        if (!closed) timer.current = window.setTimeout(connect, RECONNECT_MS);
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer.current);
      socket?.close();
    };
  }, []);

  return state;
}
