type OverlayedImageProps = {
  src: string;
  alt: string;
  overlayContent: React.ReactNode;
};

export function OverlayedImage({
  src,
  alt,
  overlayContent,
}: OverlayedImageProps) {
  return (
    <div className="relative group w-16 h-16 aspect-square rounded-md bg-white">
      <img
        src={src}
        alt={alt}
        className="w-full h-full object-cover rounded-md"
      />
      <div className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 rounded-md group-hover:opacity-100 transition-opacity">
        {overlayContent}
      </div>
    </div>
  );
}
