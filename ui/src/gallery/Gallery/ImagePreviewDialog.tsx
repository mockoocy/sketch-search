import type { IndexedImage } from "@/gallery/schema";
import { Button } from "@/general/components/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/general/components/dialog";
import { useGalleryStore } from "@/store";
import { Download, Search, Trash } from "lucide-react";
import { useState } from "react";

type ImagePreviewDialogProps = {
  image: IndexedImage;
  trigger: React.ReactNode;
};

function downloadImage(url: string, filename: string) {
  fetch(url)
    .then((response) => response.blob())
    .then((blob) => {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
      link.remove();
    });
}

export function ImagePreviewDialog({
  image,
  trigger,
}: ImagePreviewDialogProps) {
  const [open, setOpen] = useState(false);
  const setImageId = useGalleryStore((s) => s.setImageId);

  const fullSrc = `/api/images/${image.id}/view/`;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="w-1/2 p-4">
        <DialogHeader>
          <DialogTitle>{image.user_visible_name}</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          <div className="flex justify-center bg-black/5 rounded">
            <img
              src={fullSrc}
              alt={image.user_visible_name ?? image.user_visible_name}
              className="object-contain"
            />
          </div>

          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setImageId(image.id)}>
              <Search />
              Similar images
            </Button>

            <Button
              variant="secondary"
              onClick={() =>
                downloadImage(
                  fullSrc,
                  image.user_visible_name ?? `image_${image.id}`,
                )
              }
            >
              <Download />
              Download
            </Button>

            <Button variant="destructive" disabled>
              <Trash />
              Delete
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
