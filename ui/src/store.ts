import { PAGE_SIZES } from "@/gallery/Gallery/ImagesTablePagination";
import type { ImageSearchQuery } from "@/gallery/schema";
import { create } from "zustand";

type SketchSimilaritySource = {
  kind: "sketch";
  blob: Blob;
  revision: number;
};

type ImageSimilaritySource = {
  kind: "image";
  imageId: number;
};

export type SimilaritySource =
  | SketchSimilaritySource
  | ImageSimilaritySource
  | null;

type SimilaritySearchState = {
  similaritySource: SimilaritySource | null;
  setImageId: (imageId: number) => void;
  setSketch: (blob: Blob) => void;
  clearSource: () => void;
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

type GalleryStore = SimilaritySearchState & QueryState;

export const useGalleryStore = create<GalleryStore>((set) => ({
  similaritySource: null,
  setImageId: (imageId) =>
    set(() => ({
      similaritySource: { kind: "image", imageId },
    })),
  clearSource: () =>
    set(() => ({
      similaritySource: null,
    })),
  setSketch: (blob) =>
    set(() => ({
      similaritySource: { kind: "sketch", blob, revision: 0 },
    })),
  query: {
    page: 1,
    items_per_page: PAGE_SIZES[0],
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
