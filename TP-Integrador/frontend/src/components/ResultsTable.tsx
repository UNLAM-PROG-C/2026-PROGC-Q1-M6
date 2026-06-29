import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { downloadMetricsCsv } from "@/api/client";
import { useResultsSummary } from "@/hooks/useResultsSummary";
import type { ProgressState } from "@/hooks/useProgress";

interface ResultsTableProps {
  progress: ProgressState;
}

/** Tabla de resultados finales por operación con exportación a CSV (#28). */
export function ResultsTable({ progress }: ResultsTableProps) {
  const summary = useResultsSummary(progress.running, progress.current);
  const visible = !progress.running && progress.current > 0;

  if (!visible || summary.length === 0) return null;

  return (
    <Card className="mt-6">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium">
          Resultados por operación
        </CardTitle>
        <Button size="sm" variant="outline" onClick={downloadMetricsCsv}>
          Exportar CSV
        </Button>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-muted-foreground">
              <th className="py-2 text-left">Operación</th>
              <th className="py-2 text-right">CPU prom (ms)</th>
              <th className="py-2 text-right">GPU prom (ms)</th>
              <th className="py-2 text-right">Speedup</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((row) => (
              <tr key={row.operation} className="border-b last:border-0">
                <td className="py-2">{row.operation}</td>
                <td className="py-2 text-right font-mono">
                  {row.cpu_avg_ms.toFixed(2)}
                </td>
                <td className="py-2 text-right font-mono">
                  {row.gpu_avg_ms > 0 ? row.gpu_avg_ms.toFixed(2) : "—"}
                </td>
                <td className="py-2 text-right font-mono">
                  {row.speedup > 0 ? `${row.speedup.toFixed(2)}×` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
