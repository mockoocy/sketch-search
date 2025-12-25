import type { ImageSearchQuery } from "@/gallery/schema";
import { create } from "zustand";

type SketchState = {
  sketch: Blob | null;
  revision: number;
  setSketch: (blob: Blob | null) => void;
  clearSketch: () => void;
};

type QueryState = {
  query: ImageSearchQuery;
  setPage: (page: number) => void;
  setSorting: (
    order_by: ImageSearchQuery["order_by"],
    direction: ImageSearchQuery["direction"],
  ) => void;
  setFilters: (
    filters: Partial<
      Omit<
        ImageSearchQuery,
        "page" | "items_per_page" | "order_by" | "direction"
      >
    >,
  ) => void;
  setItemsPerPage: (items_per_page: number) => void;
};

type GalleryStore = SketchState & QueryState;

export const useGalleryStore = create<GalleryStore>((set) => ({
  sketch: null,
  revision: 0,
  setSketch: (blob) =>
    set((state) => ({
      sketch: blob,
      revision: state.revision + 1,
      query: { ...state.query, page: 1 },
    })),
  clearSketch: () =>
    set((state) => ({
      sketch: null,
      revision: state.revision + 1,
      query: { ...state.query, page: 1 },
    })),
  query: {
    page: 1,
    items_per_page: 10,
    order_by: "user_visible_name",
    direction: "ascending",
  },
  setPage: (page) => set((state) => ({ query: { ...state.query, page } })),
  setSorting: (order_by, direction) =>
    set((state) => ({
      query: { ...state.query, page: 1, order_by, direction },
    })),
  setFilters: (filters) =>
    set((state) => ({
      query: { ...state.query, page: 1, ...filters },
    })),
  setItemsPerPage: (items_per_page) =>
    set((state) => ({
      query: { ...state.query, page: 1, items_per_page },
    })),
}));
