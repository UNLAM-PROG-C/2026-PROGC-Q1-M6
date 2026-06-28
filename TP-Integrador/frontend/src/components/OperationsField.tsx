import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import type { Operation } from "@/api/types";

interface OperationsFieldProps {
  operations: Operation[];
  selected: string;
  onSelect: (value: string) => void;
}

/** Grilla de checkboxes de operaciones (desde /api/operations). */
export function OperationsField({
  operations,
  selected,
  onSelect,
}: OperationsFieldProps) {
  return (
    <div className="space-y-3">
      <Label>Operaciones</Label>
      <div className="grid grid-cols-2 gap-2">
        {operations.map((op) => (
          <label
            key={op.value}
            className="flex cursor-pointer items-center gap-3 rounded-lg border border-border bg-background/40 px-3 py-2.5 text-sm transition-colors hover:border-primary/50 hover:bg-secondary/60"
            onClick={() => onSelect(op.value)}
          >
            <div className={`flex h-4 w-4 items-center justify-center rounded-full border ${selected === op.value ? 'border-primary bg-primary' : 'border-primary'}`}>
              {selected === op.value && <div className="h-2 w-2 rounded-full bg-primary-foreground" />}
            </div>
            {op.label}
          </label>
        ))}
      </div>
    </div>
  );
}
