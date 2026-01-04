import {
  searchImagesResponseSchema,
  type DirectoryNode,
  type ImageSearchQuery,
  type SearchImagesResponse,
} from "@/gallery/schema";
import { apiFetch } from "@/general/api";

export async function listImages(
  query: ImageSearchQuery,
): Promise<SearchImagesResponse> {
  const url = new URL("/api/images", window.location.origin);
  const queryStrings = Object.fromEntries(
    Object.entries(query)
      .filter(([_, value]) => value !== undefined)
      .map(([key, value]) => [key, value?.toString() ?? ""]),
  );
  url.search = new URLSearchParams(queryStrings).toString();

  const data = apiFetch<SearchImagesResponse>({
    url,
    context: "List Images",
    method: "GET",
    credentials: "include",
  });
  return searchImagesResponseSchema.parse(await data);
}

type FileCreatedEvent = {
  path: string;
  created_at: string;
};

type FileDeletedEvent = {
  path: string;
};

type FileModifiedEvent = {
  path: string;
  modified_at: string;
};

type FileMovedEvent = {
  old_path: string;
  new_path: string;
  moved_at: string;
};

type FileEmbeddedEvent = {
  count: number;
  to_index: number;
};

export type FsEvent = {
  FileCreatedEvent: FileCreatedEvent;
  FileDeletedEvent: FileDeletedEvent;
  FileModifiedEvent: FileModifiedEvent;
  FileMovedEvent: FileMovedEvent;
  FileEmbeddedEvent: FileEmbeddedEvent;
};

/**
 * Simple SSE client with reference counting
 * used to store event source across multiple hook calls.
 */
class SseClient {
  private source: EventSource | null = null;
  private listeners: {
    [Key in keyof FsEvent]: Set<(event: FsEvent[Key]) => void>;
  } = {
    FileCreatedEvent: new Set(),
    FileDeletedEvent: new Set(),
    FileModifiedEvent: new Set(),
    FileMovedEvent: new Set(),
    FileEmbeddedEvent: new Set(),
  };
  private refCount = 0;
  private readonly url: string;

  constructor(url: string) {
    this.url = url;
  }

  private connect() {
    if (this.source) return;

    this.source = new EventSource(this.url);

    this.source.onerror = (error) => {
      console.error(error);
    };
  }

  private disconnect() {
    if (!this.source) return;

    this.source.close();
    this.source = null;
  }

  acquire() {
    this.connect();
    this.refCount++;
  }

  release() {
    this.refCount--;
    if (this.refCount <= 0) {
      this.disconnect();
    }
  }
  addListener<TEvent extends keyof FsEvent>(
    type: TEvent,
    callback: (event: FsEvent[TEvent]) => void,
  ): () => void {
    this.source?.addEventListener(type, (event: MessageEvent) => {
      const eventData: FsEvent[TEvent] = JSON.parse(event.data);
      callback(eventData);
    });
    this.listeners[type].add(callback);

    return () => {
      this.listeners[type].delete(callback);
    };
  }
}

export const sseFsEventsClient = new SseClient("/api/fs/events");

export type SimilaritySearchInput = {
  image: Blob;
  topK: number;
  query: ImageSearchQuery;
};

export async function similaritySearch({
  image,
  topK,
  query,
}: SimilaritySearchInput) {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("top_k", topK.toString());
  formData.append("query_json", JSON.stringify(query));
  const data = await apiFetch<SearchImagesResponse>({
    url: "/api/images/similarity-search",
    context: "Similarity Search",
    method: "POST",
    body: formData,
    credentials: "include",
  });
  return searchImagesResponseSchema.parse(await data);
}

type ImageSearchByImagePayload = {
  image_id: string;
  top_k: number;
  query: ImageSearchQuery;
};

export async function searchByImage(body: ImageSearchByImagePayload) {
  const data = await apiFetch<SearchImagesResponse>({
    url: "/api/images/search-by-image/",
    context: "Search By Image",
    method: "POST",
    body: JSON.stringify(body),
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
  });
  return searchImagesResponseSchema.parse(await data);
}

export type AddImagePayload = {
  file: File;
  directory: string;
};
export async function addImage({ file, directory }: AddImagePayload) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("directory", directory);

  return apiFetch<{ status: string }>({
    url: "/api/images/",
    method: "POST",
    body: formData,
    context: "Add Image",
    credentials: "include",
  });
}

export async function deleteImage(imageId: number) {
  return await apiFetch<void>({
    url: `/api/images/${imageId}`,
    context: "Delete Image",
    method: "DELETE",
    credentials: "include",
  });
}

export async function listDirectories() {
  return await apiFetch<DirectoryNode>({
    url: "/api/fs/watched-directories/",
    context: "List Directories",
    method: "GET",
    credentials: "include",
  });
}
