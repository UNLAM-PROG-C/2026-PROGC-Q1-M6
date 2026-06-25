import { useEffect, useState } from "react";
import { getBackend } from "@/api/client";
import type { BackendInfo } from "@/api/types";

interface BackendState {
  data: BackendInfo | null;
  loading: boolean;
  error: string | null;
}

/** Carga la información del backend activo al montar el componente. */
export function useBackend(): BackendState {
  const [state, setState] = useState<BackendState>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let active = true;
    getBackend()
      .then((data) => active && setState({ data, loading: false, error: null }))
      .catch((err: Error) =>
        active && setState({ data: null, loading: false, error: err.message }),
      );
    return () => {
      active = false;
    };
  }, []);

  return state;
}
