import { useEffect, useState } from "react";
import { getOperations } from "@/api/client";
import type { Operation } from "@/api/types";

/** Carga las operaciones disponibles (con etiquetas en español). */
export function useOperations(): Operation[] {
  const [operations, setOperations] = useState<Operation[]>([]);

  useEffect(() => {
    let active = true;
    getOperations()
      .then((data) => active && setOperations(data))
      .catch(() => active && setOperations([]));
    return () => {
      active = false;
    };
  }, []);

  return operations;
}
