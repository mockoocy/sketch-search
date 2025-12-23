import type { ImageSearchQuery, IndexedImage } from "@/gallery/schema";
import { apiFetch } from "@/general/api";

type ListImagesData = {
  images: IndexedImage[];
  total: number;
};

export async function listImages(
  query: ImageSearchQuery,
): Promise<ListImagesData> {
  const url = new URL("/api/images", window.location.origin);
  const queryStrings = Object.fromEntries(
    Object.entries(query)
      .filter(([_, value]) => value !== undefined)
      .map(([key, value]) => [key, value?.toString() ?? ""]),
  );
  url.search = new URLSearchParams(queryStrings).toString();

  return apiFetch<ListImagesData>({
    url,
    context: "List Images",
    method: "GET",
    credentials: "include",
  });
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

export type FsEvent = {
  FileCreatedEvent: FileCreatedEvent;
  FileDeletedEvent: FileDeletedEvent;
  FileModifiedEvent: FileModifiedEvent;
  FileMovedEvent: FileMovedEvent;
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

export const sseFsEventsClient = new SseClient("/api/events");

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
  const data = await apiFetch<{ images: IndexedImage[] }>({
    url: "/api/images/similarity-search",
    context: "Similarity Search",
    method: "POST",
    body: formData,
    credentials: "include",
  });
  return data.images;
}
