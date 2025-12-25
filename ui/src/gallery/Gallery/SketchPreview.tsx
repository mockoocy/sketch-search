import { Button } from "@/general/components/button";
import { useGalleryStore } from "@/store";
import { TrashIcon } from "lucide-react";

type SketchPreviewProps = {
  sketch: Blob;
};

export function SketchPreview({ sketch }: SketchPreviewProps) {
  const clearSketch = useGalleryStore((state) => state.clearSketch);

  if (!sketch) {
    return null;
  }
  const url = sketch ? URL.createObjectURL(sketch) : undefined;
  return (
    <div className="group relative w-16 h-16 rounded bg-white">
      <img
        src={url}
        alt="Sketch Preview"
        className="w-full h-full object-cover"
      />

      <div className="absolute inset-0 bg-black/50 opacity-0 rounded group-hover:opacity-100 transition-opacity" />

      <Button
        variant="destructive"
        size="icon"
        className="absolute inset-0 m-auto opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={clearSketch}
      >
        <TrashIcon className="w-4 h-4" />
      </Button>
    </div>
  );
}
