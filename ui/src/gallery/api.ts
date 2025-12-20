import type { ImageSearchQuery, IndexedImage } from "@/gallery/schema"
import { apiFetch } from "@/general/api"

type ListImagesData = {
  images: IndexedImage[]
  total: number
}

export async function listImages(query: ImageSearchQuery): Promise<ListImagesData>{
  const url = new URL("/api/images", window.location.origin)
  const queryStrings = Object.fromEntries(
    Object.entries(query).map(([key, value]) => [key, value?.toString() ?? ""])
  )
  url.search = new URLSearchParams(queryStrings).toString()

  return await apiFetch<ListImagesData>({
    url,
    context: "List Images",
    method: "GET",
    credentials: "include",
  })
}
