import z from "zod";

export const userSearchQuerySchema = z.object({
  email: z.email().optional(),
  role: z.enum(["admin", "user", "editor"]).optional(),
  page: z.number().min(1).optional(),
  pageSize: z.number().min(1).max(100).optional(),
});

export type UserSearchQuery = z.infer<typeof userSearchQuerySchema>;

export const roles = ["admin", "user", "editor"] as const;
export type UserRole = (typeof roles)[number];
export type User = {
  id: number;
  email: string;
  role: UserRole;
};
