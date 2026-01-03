import { ImagePreviewDialog } from "@/gallery/Gallery/ImagePreviewDialog";
import type { ImageRow } from "@/gallery/hooks";
import { Button } from "@/general/components/button";
import { OverlayedImage } from "@/general/components/overlayed-image";

type ThumbnailCellProps = {
  row: ImageRow;
};

export function ThumbnailCell({ row }: ThumbnailCellProps) {
  const { id, user_visible_name, directory } = row.image;

  const url = `/api/images/${id}/thumbnail/`;
  return (
    <div className="w-full flex items-center justify-between gap-2">
      <OverlayedImage
        src={url}
        alt={user_visible_name ?? "Thumbnail"}
        overlayContent={
          <ImagePreviewDialog
            image={row.image}
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
        <span className="text-muted-foreground">
          {directory} / {user_visible_name}
        </span>
      </div>
    </div>
  );
}
