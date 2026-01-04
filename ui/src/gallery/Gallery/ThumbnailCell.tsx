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
  const fullPath = `${directory}/${user_visible_name}`;

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
      <div className="w-full min-w-0 flex flex-col items-center">
        <span
          className="text-foreground truncate max-w-full"
          title={user_visible_name}
        >
          {user_visible_name}
        </span>
        <span
          className="text-muted-foreground truncate max-w-full whitespace-nowrap"
          title={fullPath}
        >
          {fullPath}
        </span>
      </div>
    </div>
  );
}
