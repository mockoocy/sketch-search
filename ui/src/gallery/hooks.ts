import { listImages } from "@/gallery/api";
import type { ImageSearchQuery } from "@/gallery/schema";
import { useQuery } from "@tanstack/react-query";


export function useListImages(query: ImageSearchQuery) {
  return useQuery({
    queryKey: [ "images" , query],
    queryFn: () => listImages(query),
  });
}
