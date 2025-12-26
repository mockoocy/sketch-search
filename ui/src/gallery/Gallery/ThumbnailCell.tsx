import { ImagePreviewDialog } from "@/gallery/Gallery/ImagePreviewDialog";
import type { IndexedImage } from "@/gallery/schema";
import { Button } from "@/general/components/button";
import { OverlayedImage } from "@/general/components/overlayed-image";

type ThumbnailCellProps = {
  image: IndexedImage;
};

export function ThumbnailCell({ image }: ThumbnailCellProps) {
  const { id, user_visible_name, path } = image;

  const url = `/api/images/${id}/thumbnail/`;
  return (
    <div className="w-full flex items-center justify-between gap-2">
      <OverlayedImage
        src={url}
        alt={user_visible_name ?? "Thumbnail"}
        overlayContent={
          <ImagePreviewDialog
            image={image}
            trigger={
              <Button variant="ghost" size="icon">
                Preview
              </Button>
            }
          />
        }
      />
      <div className="w-full flex flex-col items-center">
        <span className="text-foreground">{user_visible_name}</span>
        <span className="text-muted-foreground">{path}</span>
      </div>
    </div>
  );
}
