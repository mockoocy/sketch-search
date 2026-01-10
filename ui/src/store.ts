import type { ImageSearchQuery } from "@/gallery/schema";
import { create } from "zustand";

type SketchSimilaritySource = {
  kind: "sketch";
  blob: Blob;
};

type ImageSimilaritySource = {
  kind: "image";
  imageId: string;
};

export type SimilaritySource =
  | SketchSimilaritySource
  | ImageSimilaritySource
  | null;

type SimilaritySearchState = {
  similaritySource: SimilaritySource | null;
  setImageId: (imageId: string) => void;
  setSketch: (blob: Blob) => void;
  clearSource: () => void;
};

type QueryState = {
  sketchRevision: number;
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
  setDirectory: (directory: string | null) => void;
};

type GalleryStore = SimilaritySearchState & QueryState;

export const useGalleryStore = create<GalleryStore>((set) => ({
  similaritySource: null,
  sketchRevision: 0,
  setImageId: (imageId) =>
    set(() => ({
      similaritySource: { kind: "image", imageId },
    })),
  clearSource: () =>
    set(() => ({
      similaritySource: null,
    })),
  setSketch: (blob) =>
    set((state) => ({
      sketchRevision: state.sketchRevision + 1,
      similaritySource: { kind: "sketch", blob },
    })),
  query: {
    page: 1,
    items_per_page: 12,
    order_by: "user_visible_name",
    direction: "ascending",
    directory: ".",
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
  setDirectory: (directory) =>
    set((state) => ({
      query: { ...state.query, page: 1, directory },
    })),
}));
