import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import type { Operation } from "@/api/types";

interface OperationsFieldProps {
  operations: Operation[];
  selected: Set<string>;
  onToggle: (value: string) => void;
}

/** Grilla de checkboxes de operaciones (desde /api/operations). */
export function OperationsField({
  operations,
  selected,
  onToggle,
}: OperationsFieldProps) {
  return (
    <div className="space-y-3">
      <Label>Operaciones</Label>
      <div className="grid grid-cols-2 gap-2">
        {operations.map((op) => (
          <label
            key={op.value}
            className="flex cursor-pointer items-center gap-3 rounded-lg border border-border bg-background/40 px-3 py-2.5 text-sm transition-colors hover:border-primary/50 hover:bg-secondary/60"
          >
            <Checkbox
              checked={selected.has(op.value)}
              onCheckedChange={() => onToggle(op.value)}
            />
            {op.label}
          </label>
        ))}
      </div>
    </div>
  );
}
