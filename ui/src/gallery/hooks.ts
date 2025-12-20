import { listImages } from "@/gallery/api";
import type { ImageSearchQuery } from "@/gallery/schema";
import { useQuery } from "@tanstack/react-query";


export function useListImages(query: ImageSearchQuery) {
  const queryObject =  useQuery({
    queryKey: [ "images" , query],
    queryFn: () => listImages(query),
  });
  return queryObject;
}
