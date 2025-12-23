import { Button } from "@/general/components/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/general/components/dialog";
import { SketchCanvas, type SketchCanvasHandle } from "@/SketchCanvas";
import { useRef } from "react";

type SketchSearchDialogProps = {
  onSketchSubmit: (blob: Blob) => void;
};

export function SketchSearchDialog({
  onSketchSubmit,
}: SketchSearchDialogProps) {
  const ref = useRef<SketchCanvasHandle | null>(null);

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button type="button">Sketch search</Button>
      </DialogTrigger>

      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Draw a sketch</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          <SketchCanvas ref={ref} />

          <div className="flex gap-2">
            <Button
              type="button"
              variant="secondary"
              onClick={() => ref.current?.undo()}
            >
              Undo
            </Button>

            <Button
              type="button"
              variant="secondary"
              onClick={() => ref.current?.redo()}
            >
              Redo
            </Button>

            <Button
              type="button"
              variant="secondary"
              onClick={() => ref.current?.clear()}
            >
              Clear
            </Button>
            <Button
              type="button"
              onClick={() => {
                ref.current?.exportPng((blob) => {
                  if (blob) onSketchSubmit(blob);
                });
              }}
            >
              Submit Sketch
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
