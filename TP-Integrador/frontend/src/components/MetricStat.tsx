import type { LucideIcon } from "lucide-react";

interface MetricStatProps {
  icon: LucideIcon;
  label: string;
  value: string;
  hint?: string;
}

/** Celda de métrica (velocidad / tiempo / ETA) del panel de progreso. */
export function MetricStat({ icon: Icon, label, value, hint }: MetricStatProps) {
  return (
    <div className="rounded-lg border border-border bg-background/40 p-4">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <p className="mt-2 font-mono text-xl font-semibold tabular-nums">
        {value}
      </p>
      {hint && <p className="mt-0.5 text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}
