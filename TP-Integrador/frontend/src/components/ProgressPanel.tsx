import { Clock, Gauge, Hourglass, Radio, Zap } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { MetricStat } from "@/components/MetricStat";
import type { ProgressState } from "@/hooks/useProgress";

interface ProgressPanelProps {
  progress: ProgressState;
}

const PLACEHOLDER = "—";

/** Panel de progreso global en tiempo real (#21, #28). */
export function ProgressPanel({ progress }: ProgressPanelProps) {
  const percent = Math.round(progress.percent);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Radio className="h-4 w-4 text-primary" />
            Progreso en tiempo real
          </CardTitle>
          <ConnectionDot connected={progress.connected} />
        </div>
        <CardDescription>
          Avance global del pipeline vía WebSocket.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-end justify-between">
            <span className="font-mono text-4xl font-bold tabular-nums">
              {percent}
              <span className="text-xl text-muted-foreground">%</span>
            </span>
            <span className="font-mono text-sm tabular-nums text-muted-foreground">
              {progress.current} / {progress.total} imágenes procesadas
            </span>
          </div>
          <Progress value={progress.percent} />
        </div>

        <div className="grid grid-cols-4 gap-3">
          <MetricStat
            icon={Gauge}
            label="Velocidad"
            value={progress.speed > 0 ? progress.speed.toFixed(1) : PLACEHOLDER}
            hint="img/s"
          />
          <MetricStat
            icon={Clock}
            label="Transcurrido"
            value={progress.elapsed > 0 ? formatTime(progress.elapsed) : PLACEHOLDER}
            hint="mm:ss"
          />
          <MetricStat
            icon={Hourglass}
            label="ETA"
            value={
              progress.running && progress.eta > 0
                ? formatTime(progress.eta)
                : PLACEHOLDER
            }
            hint="estimado"
          />
          <MetricStat
            icon={Zap}
            label="Speedup"
            value={
              progress.speedup > 0
                ? `${progress.speedup.toFixed(1)}×`
                : PLACEHOLDER
            }
            hint="CPU/GPU"
          />
        </div>
      </CardContent>
    </Card>
  );
}

function formatTime(seconds: number): string {
  if (!seconds || isNaN(seconds)) return "00:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function ConnectionDot({ connected }: { connected: boolean }) {
  return (
    <span className="flex items-center gap-2 text-xs text-muted-foreground">
      <span
        className={
          connected
            ? "h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400"
            : "h-2 w-2 rounded-full bg-muted-foreground"
        }
      />
      {connected ? "En vivo" : "Desconectado"}
    </span>
  );
}
