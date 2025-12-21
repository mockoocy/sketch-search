import type { IndexedImage } from "@/gallery/schema"

type ThumbnailCellProps = {
  image: IndexedImage
}

export function ThumbnailCell({ image }: ThumbnailCellProps) {

  const {id, user_visible_name, path} = image

  const url = `/api/images/${id}/thumbnail/`
  return <div className="w-full flex items-center justify-between gap-2">
    <img src={url} alt={user_visible_name ?? "Thumbnail"} className="w-12 h-12 rounded object-cover" />
    <div className="w-full flex flex-col items-center">
      <span className="text-foreground">{user_visible_name}</span>
      <span className="text-muted-foreground">{path}</span>
    </div>
  </div>
}
