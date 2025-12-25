import { FiltersBar } from "@/gallery/Gallery/FiltersBar";
import { ImagesTable } from "@/gallery/Gallery/ImagesTable";
import {
  imageQueryKeys,
  useFsEvents,
  useImageSearch,
  type UseImageSearchOptions,
} from "@/gallery/hooks";
import { type ImageSearchQuery } from "@/gallery/schema";
import { Card, CardContent } from "@/general/components/card";
import { useGalleryStore } from "@/store";
import { useQueryClient } from "@tanstack/react-query";
import type { SortingState } from "@tanstack/react-table";
import { toast } from "sonner";

function imageOrderingToSorting(
  order_by: ImageSearchQuery["order_by"],
  direction: ImageSearchQuery["direction"],
): SortingState {
  return [
    {
      id: order_by,
      desc: direction === "descending",
    },
  ];
}
function sortingToImageOrdering(sorting: SortingState): {
  order_by: ImageSearchQuery["order_by"];
  direction: ImageSearchQuery["direction"];
} {
  const sort = sorting[0];
  const order_by = (sort?.id as ImageSearchQuery["order_by"]) || "created_at";
  const direction = sort?.desc ? "descending" : "ascending";
  return { order_by, direction };
}

export function Gallery() {
  const sketch = useGalleryStore((state) => state.sketch);
  const revision = useGalleryStore((state) => state.revision);
  const query = useGalleryStore((state) => state.query);
  const setFilters = useGalleryStore((state) => state.setFilters);
  const setSorting = useGalleryStore((state) => state.setSorting);

  const searchOptions: UseImageSearchOptions = sketch
    ? {
        searchType: "sketch",
        sketch,
        query,
        revision: revision,
      }
    : { searchType: "plain", query };

  const { data } = useImageSearch(searchOptions);

  const queryClient = useQueryClient();
  useFsEvents({
    onCreate: (event) => {
      queryClient.invalidateQueries({ queryKey: imageQueryKeys.all });
      toast.success(`File created: ${event.path}`);
    },
    onDelete: (event) => {
      queryClient.invalidateQueries({ queryKey: imageQueryKeys.all });
      toast.success(`File deleted: ${event.path}`);
    },
    onModify: (event) => {
      queryClient.invalidateQueries({ queryKey: imageQueryKeys.all });
      toast.success(`File modified: ${event.path}`);
    },
    onMove: (event) => {
      queryClient.invalidateQueries({ queryKey: imageQueryKeys.all });
      toast.success(`File moved: from ${event.old_path} to ${event.new_path}`);
    },
  });

  return (
    <Card className="w-full">
      <CardContent className="space-y-4">
        <FiltersBar onSubmit={setFilters} />

        <ImagesTable
          images={data?.images || []}
          gallerySize={data?.total || 0}
          onSortingChange={(updater) => {
            if (typeof updater === "function") {
              const currentSortingState = imageOrderingToSorting(
                query.order_by,
                query.direction,
              );
              const newSortingState = updater(currentSortingState);
              const { order_by, direction } =
                sortingToImageOrdering(newSortingState);
              setSorting(order_by, direction);
              return newSortingState;
            }
            return updater;
          }}
          sorting={imageOrderingToSorting(query.order_by, query.direction)}
        />
      </CardContent>
    </Card>
  );
}
