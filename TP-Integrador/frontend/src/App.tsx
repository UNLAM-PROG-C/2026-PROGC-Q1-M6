import { BackendIndicator } from "@/components/BackendIndicator";
import { ConfigPanel } from "@/components/ConfigPanel";
import { ProgressPanel } from "@/components/ProgressPanel";
import { ResultsGallery } from "@/components/ResultsGallery";
import { ResultsTable } from "@/components/ResultsTable";
import { SpeedupChart } from "@/components/SpeedupChart";
import { useProgress } from "@/hooks/useProgress";
import { useStartConfig } from "@/hooks/useStartConfig";

/** Layout principal del dashboard: header + dos columnas (#20, #21, #28). */
export default function App() {
  const config = useStartConfig();
  const progress = useProgress();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 border-b border-white/5 bg-background/40 backdrop-blur-2xl shadow-sm">
        <div className="container flex h-16 max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 overflow-hidden items-center justify-center rounded-lg ring-1 ring-primary/30">
              <img src="/icon.png" alt="ParallelVision Logo" className="h-full w-full object-cover" />
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

      <main className="container max-w-6xl py-8 animate-in fade-in duration-700 slide-in-from-bottom-4">
        <div className="grid gap-6 lg:grid-cols-2 items-start">
          <div className="flex flex-col gap-6 sticky top-24">
            <ConfigPanel config={config} />
          </div>
          <div className="flex flex-col gap-6">
            <ProgressPanel progress={progress} />
            
            <div className="animate-in fade-in duration-700 delay-150 slide-in-from-bottom-4 fill-mode-both">
              <SpeedupChart progress={progress} />
            </div>
            
            <div className="animate-in fade-in duration-700 delay-300 slide-in-from-bottom-4 fill-mode-both">
              <ResultsTable progress={progress} />
            </div>
          </div>
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
