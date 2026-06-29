import { FolderSearch } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { FolderPicker } from "@/components/FolderPicker";

interface FolderFieldProps {
  label: string;
  dialogTitle: string;
  value: string;
  placeholder: string;
  onSelect: (path: string) => void;
}

/** Campo de carpeta con su botón para abrir el FolderPicker. */
export function FolderField({
  label,
  dialogTitle,
  value,
  placeholder,
  onSelect,
}: FolderFieldProps) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <div className="flex items-center gap-2">
        <div className="flex min-h-10 flex-1 min-w-0 items-center break-all rounded-md border border-input bg-background/40 px-3 py-2 font-mono text-xs text-muted-foreground">
          {value || placeholder}
        </div>
        <FolderPicker title={dialogTitle} onSelect={onSelect}>
          <Button variant="outline" size="icon" aria-label={dialogTitle}>
            <FolderSearch className="h-4 w-4" />
          </Button>
        </FolderPicker>
      </div>
    </div>
  );
}
