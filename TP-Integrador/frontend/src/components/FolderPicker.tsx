import { useEffect, useState } from "react";
import { CornerLeftUp, Folder, FolderOpen, HardDrive, ArrowRight } from "lucide-react";
import { browse } from "@/api/client";
import type { BrowseResult } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface FolderPickerProps {
  title: string;
  onSelect: (path: string) => void;
  children: React.ReactNode;
}

/** Diálogo que navega el filesystem del servidor vía /api/browse (#20). */
export function FolderPicker({ title, onSelect, children }: FolderPickerProps) {
  const [open, setOpen] = useState(false);
  const [tree, setTree] = useState<BrowseResult | null>(null);
  const [inputPath, setInputPath] = useState("");

  useEffect(() => {
    if (open) {
      browse("").then(t => {
        setTree(t);
        setInputPath(t.path);
      }).catch(() => {
        setTree(null);
      });
    }
  }, [open]);

  const navigate = (path: string) =>
    browse(path).then(t => {
      setTree(t);
      setInputPath(t.path);
    }).catch(() => setTree(null));

  const confirm = () => {
    if (tree) onSelect(tree.path);
    setOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      navigate(inputPath);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription className="sr-only">Seleccionar carpeta del sistema</DialogDescription>
        </DialogHeader>
        
        <div className="flex items-center gap-2 mt-2">
          <HardDrive className="h-4 w-4 shrink-0 text-muted-foreground" />
          <input 
            type="text" 
            value={inputPath}
            onChange={(e) => setInputPath(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex h-9 w-full rounded-md border border-input bg-background/40 px-3 py-1 font-mono text-xs shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            placeholder="Ruta de la carpeta..."
          />
          <Button variant="outline" size="icon" onClick={() => navigate(inputPath)} className="h-9 w-9 shrink-0">
            <ArrowRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="h-72 space-y-1 overflow-y-auto rounded-lg border border-border bg-background/40 p-2">
          {tree?.parent && (
            <button
              onClick={() => navigate(tree.parent as string)}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-secondary text-left"
            >
              <CornerLeftUp className="h-4 w-4 shrink-0" />
              <span className="break-all">.. (carpeta superior)</span>
            </button>
          )}
          {tree?.entries.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-muted-foreground">
              No hay subcarpetas aquí.
            </p>
          )}
          {tree?.entries.map((entry) => (
            <button
              key={entry.path}
              onClick={() => navigate(entry.path)}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-secondary"
            >
              <Folder className="h-4 w-4 shrink-0 text-primary" />
              <span className="break-all">{entry.name}</span>
            </button>
          ))}
        </div>

        <DialogFooter>
          <Button onClick={confirm} disabled={!tree}>
            <FolderOpen className="h-4 w-4" />
            Elegir esta carpeta
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
