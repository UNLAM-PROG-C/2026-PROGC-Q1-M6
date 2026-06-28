import { useState, useRef } from "react";

interface BeforeAfterSliderProps {
  originalSrc: string;
  transformedSrc: string;
}

export function BeforeAfterSlider({ originalSrc, transformedSrc }: BeforeAfterSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percent);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (e.buttons !== 1) return; // Only if left mouse button is pressed
    handleMove(e.clientX);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    handleMove(e.touches[0].clientX);
  };

  return (
    <div 
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none touch-none bg-black/50 rounded-lg flex items-center justify-center"
      onMouseMove={handleMouseMove}
      onTouchMove={handleTouchMove}
      onClick={(e) => handleMove(e.clientX)}
    >
      {/* Background Image (Transformed) */}
      <img
        src={transformedSrc}
        alt="Transformed"
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
      />

      {/* Foreground Image (Original) with Clip Path */}
      <img
        src={originalSrc}
        alt="Original"
        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
      />

      {/* Slider Handle */}
      <div
        className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize flex items-center justify-center shadow-[0_0_10px_rgba(0,0,0,0.5)]"
        style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
      >
        <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg border border-neutral-200">
          <div className="flex gap-1">
            <div className="w-0.5 h-3 bg-neutral-400 rounded-full" />
            <div className="w-0.5 h-3 bg-neutral-400 rounded-full" />
          </div>
        </div>
      </div>
      
      {/* Labels */}
      <div className="absolute bottom-4 left-4 bg-black/70 text-white text-xs px-2 py-1 rounded shadow-md pointer-events-none">
        Original
      </div>
      <div className="absolute bottom-4 right-4 bg-black/70 text-white text-xs px-2 py-1 rounded shadow-md pointer-events-none">
        Procesada
      </div>
    </div>
  );
}
