import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useSpeedupChart } from "@/hooks/useSpeedupChart";
import type { ProgressState } from "@/hooks/useProgress";

interface SpeedupChartProps {
  progress: ProgressState;
}

/** Gráfico de speedup CPU vs GPU en tiempo real (#28). */
export function SpeedupChart({ progress }: SpeedupChartProps) {
  const data = useSpeedupChart(progress.running, progress.current > 0);

  if (data.length === 0) return null;

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle className="text-sm font-medium">
          Speedup CPU vs GPU — tiempos por lote
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data}>
            <XAxis
              dataKey="batch"
              label={{ value: "Lote", position: "insideBottom", offset: -2 }}
            />
            <YAxis
              label={{ value: "ms", angle: -90, position: "insideLeft" }}
            />
            <Tooltip formatter={(v: number) => `${v.toFixed(2)} ms`} />
            <Legend />
            <Line
              type="monotone"
              dataKey="cpu_ms"
              name="CPU"
              stroke="#ef4444"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="gpu_ms"
              name="GPU"
              stroke="#3b82f6"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
