import { Button } from "@/general/components/button";
import { OverlayedImage } from "@/general/components/overlayed-image";
import { useGalleryStore } from "@/store";
import { TrashIcon } from "lucide-react";

type SketchVariantProps = {
  variant: "sketch";
  sketch: Blob;
};

type ImageVariantProps = {
  variant: "image";
  imageId: string;
};

export type ImageFilterPreviewProps = SketchVariantProps | ImageVariantProps;

export function ImageFilterPreview(props: ImageFilterPreviewProps) {
  const clearSource = useGalleryStore((state) => state.clearSource);

  const url =
    props.variant === "sketch"
      ? URL.createObjectURL(props.sketch)
      : `/api/images/${props.imageId}/thumbnail/`;
  return (
    <OverlayedImage
      src={url}
      alt={props.variant === "sketch" ? "Sketch Preview" : "Image Preview"}
      overlayContent={
        <Button
          onClick={clearSource}
          variant="ghost"
          size="icon"
          className="w-full h-full text-white"
        >
          <TrashIcon />
        </Button>
      }
    />
  );
}
