import {
  listImages,
  similaritySearch,
  sseFsEventsClient,
  type FsEvent,
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

type UseSimilaritySearchOptions = {
  searchType: "sketch";
  sketch: Blob;
  revision: number;
  query: ImageSearchQuery;
};

type UseListImagesOptions = {
  searchType: "plain";
  query: ImageSearchQuery;
};

export type UseImageSearchOptions =
  | UseListImagesOptions
  | UseSimilaritySearchOptions;
export function useImageSearch(options: UseImageSearchOptions) {
  const queryKey =
    options.searchType === "plain"
      ? imageQueryKeys.list(options.query)
      : imageQueryKeys.similaritySearch(options.query, options.revision);

  const queryFn =
    options.searchType === "plain"
      ? () => listImages(options.query)
      : () =>
          similaritySearch({
            image: options.sketch!,
            topK: options.query.items_per_page,
            query: options.query,
          });
  return useQuery({ queryKey, queryFn });
}
