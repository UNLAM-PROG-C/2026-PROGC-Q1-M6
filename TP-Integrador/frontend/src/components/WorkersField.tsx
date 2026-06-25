import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

const MIN_WORKERS = 1;
const MAX_WORKERS = 16;

interface WorkersFieldProps {
  workers: number;
  onChange: (count: number) => void;
}

/** Control deslizante de la cantidad de hilos worker (1–16). */
export function WorkersField({ workers, onChange }: WorkersFieldProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label>Hilos de procesamiento</Label>
        <span className="rounded-md bg-secondary px-2.5 py-1 font-mono text-sm font-semibold tabular-nums text-primary">
          {workers}
        </span>
      </div>
      <Slider
        min={MIN_WORKERS}
        max={MAX_WORKERS}
        step={1}
        value={[workers]}
        onValueChange={(values) => onChange(values[0])}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{MIN_WORKERS} hilo</span>
        <span>{MAX_WORKERS} hilos</span>
      </div>
    </div>
  );
}
