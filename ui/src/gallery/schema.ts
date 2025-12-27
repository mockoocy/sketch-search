import { z } from "zod";

export const imageSearchQuerySchema = z.object({
  page: z.number().int().min(1).default(1),
  items_per_page: z.number().int().min(1).max(50).default(10),
  order_by: z.enum(["created_at", "modified_at", "user_visible_name"]),
  direction: z.enum(["ascending", "descending"]),
  name_contains: z.string().min(1).optional(),
  created_min: z.iso.datetime().optional(),
  created_max: z.iso.datetime().optional(),
  modified_min: z.iso.datetime().optional(),
  modified_max: z.iso.datetime().optional(),
  directory: z.string().min(1).nullable().default(null),
});

export type ImageSearchQueryInput = z.input<typeof imageSearchQuerySchema>;
export type ImageSearchQuery = z.output<typeof imageSearchQuerySchema>;

export type IndexedImage = {
  id: number;
  path: string;
  user_visible_name: string;
  created_at?: string;
  modified_at?: string;
};

export type DirectoryNode = {
  path: string;
  parent?: string;
  created_at: string;
  modified_at: string;
  children: DirectoryNode[];
};

// for some reason zod compiles date types to string
// when using datetime(). So an extra type is needed.
export type Filters = Pick<
  ImageSearchQuery,
  | "name_contains"
  | "created_min"
  | "created_max"
  | "modified_min"
  | "modified_max"
>;
