import { useEffect, useState } from "react";
import { CornerLeftUp, Folder, FolderOpen, HardDrive } from "lucide-react";
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

  useEffect(() => {
    if (open) browse("").then(setTree).catch(() => setTree(null));
  }, [open]);

  const navigate = (path: string) =>
    browse(path).then(setTree).catch(() => setTree(null));

  const confirm = () => {
    if (tree) onSelect(tree.path);
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription className="truncate font-mono text-xs">
            <HardDrive className="mr-1 inline h-3 w-3" />
            {tree?.path ?? "Cargando…"}
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-72 space-y-1 overflow-y-auto rounded-lg border border-border bg-background/40 p-2">
          {tree?.parent && (
            <button
              onClick={() => navigate(tree.parent as string)}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-secondary"
            >
              <CornerLeftUp className="h-4 w-4" />
              .. (carpeta superior)
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
              <Folder className="h-4 w-4 text-primary" />
              {entry.name}
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
