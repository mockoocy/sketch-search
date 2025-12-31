import type { User, UserSearchQuery } from "@/admin/schema";
import { apiFetch } from "@/general/api";

type ListUsersResponse = {
  total: number;
  users: User[];
};

export function listUsers(query: UserSearchQuery) {
  const url = new URL("/api/users", window.location.origin);
  const queryStrings = Object.fromEntries(
    Object.entries(query)
      .filter(([_, value]) => value !== undefined)
      .map(([key, value]) => [key, value?.toString() ?? ""]),
  );
  url.search = new URLSearchParams(queryStrings).toString();
  return apiFetch<ListUsersResponse>({
    url,
    context: "List Users",
    method: "GET",
    credentials: "include",
  });
}

export function addUser(user: Omit<User, "id">) {
  return apiFetch<User>({
    url: "/api/users",
    context: "Add User",
    method: "POST",
    body: JSON.stringify(user),
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
  });
}
export type UpdateUserOptions = {
  userId: number;
  user: User;
};

export function updateUser({ userId, user }: UpdateUserOptions) {
  return apiFetch<User>({
    url: `/api/users/${encodeURIComponent(userId)}`,
    context: "Update User",
    method: "PUT",
    body: JSON.stringify(user),
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
  });
}

export function deleteUser(userId: string) {
  return apiFetch<void>({
    url: `/api/users/${encodeURIComponent(userId)}`,
    context: "Delete User",
    method: "DELETE",
    credentials: "include",
  });
}
