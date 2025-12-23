import { listImages, sseFsEventsClient, type FsEvent } from "@/gallery/api";
import type { ImageSearchQuery } from "@/gallery/schema";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";

export const imageQueryKeys = {
  all: ["images"] as const,
  lists: () => [...imageQueryKeys.all, "list"] as const,
  list: (query: ImageSearchQuery) =>
    [...imageQueryKeys.lists(), query] as const,
} as const;

export function useListImages(query: ImageSearchQuery) {
  return useQuery({
    queryKey: imageQueryKeys.list(query),
    queryFn: () => listImages(query),
  });
}

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
