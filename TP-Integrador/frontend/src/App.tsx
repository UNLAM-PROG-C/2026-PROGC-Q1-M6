import { Boxes } from "lucide-react";
import { BackendIndicator } from "@/components/BackendIndicator";
import { ConfigPanel } from "@/components/ConfigPanel";
import { ProgressPanel } from "@/components/ProgressPanel";
import { ResultsGallery } from "@/components/ResultsGallery";
import { useStartConfig } from "@/hooks/useStartConfig";

/** Layout principal del dashboard: header + dos columnas (#20, #21). */
export default function App() {
  const config = useStartConfig();

  return (
    <div className="min-h-screen">
      <header className="border-b border-border/70 bg-card/30 backdrop-blur-md">
        <div className="container flex h-16 max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15 text-primary ring-1 ring-primary/30">
              <Boxes className="h-5 w-5" />
            </span>
            <div className="leading-tight">
              <h1 className="text-sm font-semibold tracking-tight">
                ParallelVision
              </h1>
              <p className="text-xs text-muted-foreground">
                Procesamiento concurrente de imágenes
              </p>
            </div>
          </div>
          <BackendIndicator />
        </div>
      </header>

      <main className="container max-w-6xl py-8">
        <div className="grid gap-6 lg:grid-cols-2">
          <ConfigPanel config={config} />
          <ProgressPanel />
        </div>
        
        <ResultsGallery 
          inputDir={config.inputDir} 
          outputDir={config.outputDir} 
        />
        
        <footer className="mt-10 text-center text-xs text-muted-foreground">
          ParallelVision · UNLaM Programación Concurrente · Capa 5 (Dashboard)
        </footer>
      </main>
    </div>
  );
}
