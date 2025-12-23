import { FiltersBar } from "@/gallery/Gallery/FiltersBar";
import { ImagesTable } from "@/gallery/Gallery/ImagesTable";
import { PAGE_SIZES } from "@/gallery/Gallery/ImagesTablePagination";
import { queryReducer } from "@/gallery/Gallery/queryReducer";
import {
  imageQueryKeys,
  useFsEvents,
  useSimilaritySearch,
} from "@/gallery/hooks";
import { type Filters, type ImageSearchQuery } from "@/gallery/schema";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/general/components/card";
import { SketchSearchDialog } from "@/SketchCanvas/SketchSearchDialog";
import { useQueryClient } from "@tanstack/react-query";
import type { SortingState } from "@tanstack/react-table";
import { useReducer } from "react";
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
  const [query, dispatch] = useReducer(queryReducer, {
    page: 1,
    items_per_page: PAGE_SIZES[0],
    order_by: "created_at",
    direction: "descending",
  });

  const { data, mutate: similaritySearch } = useSimilaritySearch();

  console.log({ data });

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

  const onFilterSubmit = (filters: Filters) => {
    dispatch({ type: "setFilters", filters });
  };

  const onItemsPerPageChange = (items_per_page: number) => {
    dispatch({ type: "setItemsPerPage", items_per_page });
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Images</CardTitle>
        <SketchSearchDialog
          onSketchSubmit={(blob) => {
            similaritySearch({ image: blob, topK: 10, query });
          }}
        />
      </CardHeader>

      <CardContent className="space-y-4">
        <FiltersBar onSubmit={onFilterSubmit} />

        <ImagesTable
          query={query}
          onSortingChange={(updater) => {
            dispatch({ type: "setPage", page: 1 });
            if (typeof updater === "function") {
              const currentSortingState = imageOrderingToSorting(
                query.order_by,
                query.direction,
              );
              const newSortingState = updater(currentSortingState);
              dispatch({
                type: "setSorting",
                ...sortingToImageOrdering(newSortingState),
              });
              return newSortingState;
            }
            return updater;
          }}
          sorting={imageOrderingToSorting(query.order_by, query.direction)}
          onPageChange={(page) => dispatch({ type: "setPage", page })}
          onItemsPerPageChange={onItemsPerPageChange}
        />
      </CardContent>
    </Card>
  );
}
