import {
  addImage,
  deleteImage,
  listDirectories,
  listImages,
  searchByImage,
  similaritySearch,
  sseFsEventsClient,
  type AddImagePayload,
  type FsEvent,
} from "@/gallery/api";
import type {
  DirectoryNode,
  ImageSearchQuery,
  IndexedImage,
  SearchImagesResponse,
} from "@/gallery/schema";
import { findNodeByPath } from "@/gallery/utils";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

async function hashBlob(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const hash = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export const imageQueryKeys = {
  all: ["images"] as const,
  lists: () => [...imageQueryKeys.all, "list"] as const,
  list: (query: ImageSearchQuery) =>
    [...imageQueryKeys.lists(), query] as const,
  similaritySearch: (query: ImageSearchQuery, revision: number) =>
    [...imageQueryKeys.list(query), "similarity", revision] as const,
  searchByImage: (query: ImageSearchQuery, imageId: string) =>
    [...imageQueryKeys.list(query), "search-by-image", imageId] as const,
} as const;

export type UseFsEventsOptions = {
  onCreate: (event: FsEvent["FileCreatedEvent"]) => void;
  onDelete: (event: FsEvent["FileDeletedEvent"]) => void;
  onModify: (event: FsEvent["FileModifiedEvent"]) => void;
  onMove: (event: FsEvent["FileMovedEvent"]) => void;
  onEmbedded: (event: FsEvent["FileEmbeddedEvent"]) => void;
};

export function useFsEvents({
  onCreate,
  onDelete,
  onModify,
  onMove,
  onEmbedded,
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
    if (onEmbedded) {
      removers.push(
        sseFsEventsClient.addListener("FileEmbeddedEvent", onEmbedded),
      );
    }
    return () => {
      removers.forEach((remove) => remove());
      sseFsEventsClient.release();
    };
  }, [onCreate, onDelete, onModify, onMove, onEmbedded]);
}

type UseSketchSearchOptions = {
  searchType: "sketch";
  sketch: Blob;
  revision: number;
  query: ImageSearchQuery;
};

type UseImageSimilaritySearchOptions = {
  searchType: "image";
  imageId: string;
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
function useImageSearch(options: UseImageSearchOptions) {
  let queryKey: readonly unknown[];
  let queryFn: () => Promise<SearchImagesResponse>;

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
      hashBlob(options.sketch).then((hash) =>
        console.log("Sketch hash:", hash),
      );
      console.log({ revision: options.revision });
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

export function useAddImage() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: AddImagePayload) => addImage(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: imageQueryKeys.lists() });
    },
  });
}

export function useDeleteImage() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (imageId: string) => deleteImage(imageId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: imageQueryKeys.lists() });
    },
  });
}

function useListDirectories() {
  return useQuery({
    queryKey: ["directories"],
    queryFn: listDirectories,
  });
}

export type DirectoryRow = { kind: "directory"; directory: DirectoryNode };
export type ImageRow = { kind: "image"; image: IndexedImage };
export type BackRow = { kind: "back"; parentDirectory: DirectoryNode };

export type GalleryRow = DirectoryRow | ImageRow | BackRow;

export function useGetGalleryRows(searchOptions: UseImageSearchOptions) {
  const imagesQuery = useImageSearch(searchOptions);
  const directoriesQuery = useListDirectories();

  const isLoading = imagesQuery.isLoading || directoriesQuery.isLoading;
  const error = imagesQuery.error || directoriesQuery.error;
  const rows: GalleryRow[] = [];
  if (isLoading || error || !imagesQuery.data || !directoriesQuery.data) {
    return {
      isLoading,
      error,
      rows,
      gallerySize: 0,
      changedImagesCount: 0,
    };
  }

  const { directory: directoryPath } = searchOptions.query;

  if (directoryPath === null) {
    for (const image of imagesQuery.data.images) {
      rows.push({ kind: "image", image });
    }

    return {
      isLoading,
      error,
      rows,
      gallerySize: imagesQuery.data.total || 0,
    };
  }

  const directories = directoriesQuery.data;
  const relativeRoot = findNodeByPath(directories, directoryPath);

  if (relativeRoot === null) {
    return {
      isLoading,
      error: new Error("Directory not found"),
      rows,
      gallerySize: 0,
    };
  }

  const parentDirectory = findNodeByPath(directories, relativeRoot.parent);
  if (parentDirectory) {
    rows.push({ kind: "back", parentDirectory });
  }

  for (const childDir of relativeRoot.children) {
    rows.push({ kind: "directory", directory: childDir });
  }

  for (const image of imagesQuery.data.images) {
    rows.push({ kind: "image", image });
  }

  return {
    isLoading,
    error,
    rows,
    gallerySize: imagesQuery.data.total || 0,
  };
}
