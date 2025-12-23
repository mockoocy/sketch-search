import type { Filters, ImageSearchQuery } from "@/gallery/schema";

type Action =
  | { type: "setPage"; page: number }
  | {
      type: "setSorting";
      order_by: ImageSearchQuery["order_by"];
      direction: ImageSearchQuery["direction"];
    }
  | ({ type: "setFilters" } & { filters: Filters })
  | { type: "setItemsPerPage"; items_per_page: number };

export function queryReducer(
  state: ImageSearchQuery,
  action: Action,
): ImageSearchQuery {
  switch (action.type) {
    case "setPage":
      return { ...state, page: action.page };
    case "setSorting":
      return {
        ...state,
        page: 1,
        order_by: action.order_by,
        direction: action.direction,
      };
    case "setFilters":
      return { ...state, page: 1, ...action.filters };
    case "setItemsPerPage":
      return { ...state, page: 1, items_per_page: action.items_per_page };
  }
}
