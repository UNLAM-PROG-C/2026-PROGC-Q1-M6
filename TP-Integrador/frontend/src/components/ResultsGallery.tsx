import { useState, useEffect, useRef } from "react";
import { Image, Search, RefreshCw, FolderOpen } from "lucide-react";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { ImageInspector } from "./ImageInspector";
import { useProgress } from "@/hooks/useProgress";

interface ResultItem {
  original_filename: string;
  transformed_filename: string;
  operation: string;
}

interface ResultsGalleryProps {
  inputDir: string;
  outputDir: string;
}

export function ResultsGallery({ inputDir, outputDir }: ResultsGalleryProps) {
  const [results, setResults] = useState<ResultItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  const fetchResults = async () => {
    if (!outputDir) return;
    
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/results?output_dir=${encodeURIComponent(outputDir)}`);
      if (!res.ok) {
        throw new Error("No se pudo cargar el archivo manifest.json. ¿Finalizó el procesamiento?");
      }
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (outputDir) {
      fetchResults();
    }
  }, [outputDir]);

  const progress = useProgress();
  const prevRunningRef = useRef(progress.running);

  useEffect(() => {
    if (prevRunningRef.current === true && progress.running === false) {
      // Auto-refresh when pipeline finishes
      fetchResults();
    }
    prevRunningRef.current = progress.running;
  }, [progress.running]);

  const openInspector = (index: number) => {
    setCurrentIndex(index);
    setInspectorOpen(true);
  };

  if (!outputDir) {
    return null;
  }

  return (
    <>
      <Card className="mt-6">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Image className="w-5 h-5 text-primary" />
              Galería de Resultados
            </CardTitle>
            <CardDescription>
              Imágenes procesadas en: <span className="font-mono text-xs">{outputDir}</span>
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={fetchResults} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Actualizar
          </Button>
        </CardHeader>
        <CardContent>
          {error && (
            <div className="p-4 mb-4 text-sm text-destructive-foreground bg-destructive/20 rounded-lg border border-destructive/30">
              {error}
            </div>
          )}
          
          {results.length === 0 && !loading && !error ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground border-2 border-dashed rounded-lg">
              <FolderOpen className="w-12 h-12 mb-4 opacity-20" />
              <p>No se encontraron resultados en esta carpeta.</p>
              <p className="text-xs mt-1">Asegurate de iniciar un procesamiento primero.</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {results.map((item, idx) => {
                const thumbnailUrl = `/api/image?dir_path=${encodeURIComponent(outputDir)}&filename=${encodeURIComponent(item.transformed_filename)}`;
                return (
                  <div 
                    key={idx} 
                    className="group relative aspect-square rounded-lg overflow-hidden border bg-black/5 cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                    onClick={() => openInspector(idx)}
                  >
                    <img 
                      src={thumbnailUrl} 
                      alt={item.transformed_filename}
                      className="w-full h-full object-cover transition-transform group-hover:scale-105"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <Search className="w-8 h-8 text-white drop-shadow-md" />
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 p-2 transform translate-y-full group-hover:translate-y-0 transition-transform">
                      <p className="text-[10px] text-white truncate" title={item.original_filename}>
                        {item.original_filename}
                      </p>
                      <span className="inline-block mt-1 text-[9px] px-1.5 py-0.5 rounded bg-primary/20 text-primary-foreground border border-primary/30">
                        {item.operation}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      <ImageInspector
        isOpen={inspectorOpen}
        onClose={() => setInspectorOpen(false)}
        results={results}
        currentIndex={currentIndex}
        onNext={() => setCurrentIndex((prev) => Math.min(results.length - 1, prev + 1))}
        onPrev={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
        outputDir={outputDir}
        inputDir={inputDir}
      />
    </>
  );
}
