import {
  listImages,
  searchByImage,
  similaritySearch,
  sseFsEventsClient,
  type FsEvent,
  type ListImagesData,
} from "@/gallery/api";
import type { ImageSearchQuery } from "@/gallery/schema";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";

export const imageQueryKeys = {
  all: ["images"] as const,
  lists: () => [...imageQueryKeys.all, "list"] as const,
  list: (query: ImageSearchQuery) =>
    [...imageQueryKeys.lists(), query] as const,
  similaritySearch: (query: ImageSearchQuery, revision: number) =>
    [...imageQueryKeys.list(query), "similarity", revision] as const,
  searchByImage: (query: ImageSearchQuery, imageId: number) =>
    [...imageQueryKeys.list(query), "search-by-image", imageId] as const,
} as const;

export type UseFsEventsOptions = {
  onCreate: (event: FsEvent["FileCreatedEvent"]) => void;
  onDelete: (event: FsEvent["FileDeletedEvent"]) => void;
  onModify: (event: FsEvent["FileModifiedEvent"]) => void;
  onMove: (event: FsEvent["FileMovedEvent"]) => void;
};

export function useFsEvents({
  onCreate,
  onDelete,
  onModify,
  onMove,
}: Partial<UseFsEventsOptions>) {
  useEffect(() => {
    sseFsEventsClient.acquire();
    const removers: (() => void)[] = [];
    if (onCreate) {
      removers.push(
        sseFsEventsClient.addListener("FileCreatedEvent", onCreate),
      );
    }
    if (onDelete) {
      removers.push(
        sseFsEventsClient.addListener("FileDeletedEvent", onDelete),
      );
    }
    if (onModify) {
      removers.push(
        sseFsEventsClient.addListener("FileModifiedEvent", onModify),
      );
    }
    if (onMove) {
      removers.push(sseFsEventsClient.addListener("FileMovedEvent", onMove));
    }
    return () => {
      removers.forEach((remove) => remove());
      sseFsEventsClient.release();
    };
  }, [onCreate, onDelete, onModify, onMove]);
}

type UseSketchSearchOptions = {
  searchType: "sketch";
  sketch: Blob;
  revision: number;
  query: ImageSearchQuery;
};

type UseImageSimilaritySearchOptions = {
  searchType: "image";
  imageId: number;
  query: ImageSearchQuery;
};

type UseListImagesOptions = {
  searchType: "plain";
  query: ImageSearchQuery;
};

export type UseImageSearchOptions =
  | UseListImagesOptions
  | UseImageSimilaritySearchOptions
  | UseSketchSearchOptions;
export function useImageSearch(options: UseImageSearchOptions) {
  let queryKey: readonly unknown[];
  let queryFn: () => Promise<ListImagesData>;

  switch (options.searchType) {
    case "plain":
      queryKey = imageQueryKeys.list(options.query);
      queryFn = () => listImages(options.query);
      break;
    case "image":
      queryKey = imageQueryKeys.searchByImage(options.query, options.imageId);
      queryFn = () =>
        searchByImage({
          image_id: options.imageId,
          query: options.query,
          top_k: options.query.items_per_page,
        });
      break;
    case "sketch":
      queryKey = imageQueryKeys.similaritySearch(
        options.query,
        options.revision,
      );
      queryFn = () =>
        similaritySearch({
          image: options.sketch,
          query: options.query,
          topK: options.query.items_per_page,
        });
      break;
  }

  return useQuery({ queryKey, queryFn });
}
