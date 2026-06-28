import { X, ChevronLeft, ChevronRight } from "lucide-react";
import { BeforeAfterSlider } from "./BeforeAfterSlider";
import { Button } from "./ui/button";

interface ResultItem {
  original_filename: string;
  transformed_filename: string;
  operation: string;
}

interface ImageInspectorProps {
  isOpen: boolean;
  onClose: () => void;
  results: ResultItem[];
  currentIndex: number;
  onNext: () => void;
  onPrev: () => void;
  outputDir: string;
  inputDir: string;
}

export function ImageInspector({
  isOpen,
  onClose,
  results,
  currentIndex,
  onNext,
  onPrev,
  outputDir,
  inputDir,
}: ImageInspectorProps) {
  if (!isOpen || results.length === 0) return null;

  const currentResult = results[currentIndex];
  
  // Construir las URLs de la API para obtener las imágenes
  const originalUrl = `/api/image?dir_path=${encodeURIComponent(inputDir)}&filename=${encodeURIComponent(currentResult.original_filename)}`;
  const transformedUrl = `/api/image?dir_path=${encodeURIComponent(outputDir)}&filename=${encodeURIComponent(currentResult.transformed_filename)}`;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background/95 backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border/50">
        <div className="flex flex-col">
          <h2 className="text-lg font-semibold">
            {currentResult.original_filename}
          </h2>
          <span className="text-sm text-muted-foreground">
            Operación: {currentResult.operation} • {currentIndex + 1} de {results.length}
          </span>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="w-6 h-6" />
        </Button>
      </div>

      {/* Main Content */}
      <div className="flex-1 relative flex items-center justify-center p-4 md:p-8 overflow-hidden">
        {/* Nav buttons */}
        <Button 
          variant="outline" 
          size="icon" 
          className="absolute left-4 z-10 w-12 h-12 rounded-full shadow-lg"
          onClick={onPrev}
          disabled={currentIndex === 0}
        >
          <ChevronLeft className="w-8 h-8" />
        </Button>

        <div className="w-full h-full max-w-6xl mx-auto bg-black rounded-lg border shadow-xl">
          <BeforeAfterSlider
            originalSrc={originalUrl}
            transformedSrc={transformedUrl}
          />
        </div>

        <Button 
          variant="outline" 
          size="icon" 
          className="absolute right-4 z-10 w-12 h-12 rounded-full shadow-lg"
          onClick={onNext}
          disabled={currentIndex === results.length - 1}
        >
          <ChevronRight className="w-8 h-8" />
        </Button>
      </div>
    </div>
  );
}
