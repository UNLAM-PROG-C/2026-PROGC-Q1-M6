import { useState } from "react";
import { Play, Settings2 } from "lucide-react";
import { start } from "@/api/client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FolderField } from "@/components/FolderField";
import { OperationsField } from "@/components/OperationsField";
import { WorkersField } from "@/components/WorkersField";
import { useOperations } from "@/hooks/useOperations";
import { type StartConfigForm } from "@/hooks/useStartConfig";

/** Panel de configuración del procesamiento (#20). */
export function ConfigPanel({ config }: { config: StartConfigForm }) {
  const operations = useOperations();
  const [status, setStatus] = useState<string | null>(null);

  const handleStart = async () => {
    setStatus(null);
    try {
      const res = await start(config.toPayload());
      setStatus(res.message);
    } catch (err) {
      setStatus((err as Error).message);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings2 className="h-4 w-4 text-primary" />
          Configuración
        </CardTitle>
        <CardDescription>
          Elegí carpetas, operaciones e hilos antes de iniciar.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <FolderField
          label="Carpeta de entrada"
          dialogTitle="Seleccionar carpeta de entrada"
          placeholder="Sin seleccionar…"
          value={config.inputDir}
          onSelect={config.setInputDir}
        />
        <FolderField
          label="Carpeta de salida"
          dialogTitle="Seleccionar carpeta de salida"
          placeholder="Sin seleccionar…"
          value={config.outputDir}
          onSelect={config.setOutputDir}
        />
        <OperationsField
          operations={operations}
          selected={config.operation}
          onSelect={config.setOperation}
        />
        <WorkersField workers={config.workers} onChange={config.setWorkers} />

        <div className="space-y-3 pt-2">
          <Button
            size="lg"
            className="w-full"
            disabled={!config.ready}
            onClick={handleStart}
          >
            <Play className="h-4 w-4" />
            Iniciar procesamiento
          </Button>
          {!config.ready && (
            <p className="text-center text-xs text-muted-foreground">
              Seleccioná ambas carpetas y al menos una operación.
            </p>
          )}
          {status && (
            <p className="rounded-lg border border-primary/30 bg-accent/40 px-3 py-2 text-center text-xs text-accent-foreground">
              {status}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
