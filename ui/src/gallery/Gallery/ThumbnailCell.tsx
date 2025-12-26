import type { IndexedImage } from "@/gallery/schema";
import { Button } from "@/general/components/button";
import { OverlayedImage } from "@/general/components/overlayed-image";
import { useGalleryStore } from "@/store";
import { Search } from "lucide-react";

type ThumbnailCellProps = {
  image: IndexedImage;
};

export function ThumbnailCell({ image }: ThumbnailCellProps) {
  const { id, user_visible_name, path } = image;

  const setImageId = useGalleryStore((state) => state.setImageId);

  const url = `/api/images/${id}/thumbnail/`;
  return (
    <div className="w-full flex items-center justify-between gap-2">
      <OverlayedImage
        src={url}
        alt={user_visible_name}
        overlayContent={
          <Button
            onClick={() => setImageId(id)}
            variant="ghost"
            size="icon"
            className="w-full h-full text-white"
          >
            <Search />
          </Button>
        }
      />
      <div className="w-full flex flex-col items-center">
        <span className="text-foreground">{user_visible_name}</span>
        <span className="text-muted-foreground">{path}</span>
      </div>
    </div>
  );
}
