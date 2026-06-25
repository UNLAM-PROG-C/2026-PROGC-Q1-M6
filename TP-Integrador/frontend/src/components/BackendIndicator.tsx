import { Cpu, Loader2, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useBackend } from "@/hooks/useBackend";

/** Badge con el backend de procesamiento detectado (#20). */
export function BackendIndicator() {
  const { data, loading, error } = useBackend();

  if (loading) {
    return (
      <Badge variant="default">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Detectando backend…
      </Badge>
    );
  }

  if (error || !data) {
    return (
      <Badge variant="cpu">
        <Cpu className="h-3.5 w-3.5" />
        Backend no disponible
      </Badge>
    );
  }

  const Icon = data.is_gpu ? Zap : Cpu;
  return (
    <Badge variant={data.is_gpu ? "gpu" : "cpu"}>
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-pulse-ring rounded-full bg-current opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
      </span>
      <Icon className="h-3.5 w-3.5" />
      <span className="font-semibold">{data.backend_name}</span>
      <span className="text-current/70">· {data.device_info}</span>
    </Badge>
  );
}
