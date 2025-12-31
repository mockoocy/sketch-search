import {
  addUser,
  deleteUser,
  listUsers,
  updateUser,
  type UpdateUserOptions,
} from "@/admin/api";
import type { User, UserSearchQuery } from "@/admin/schema";
import { useCurrentSession } from "@/auth/hooks";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

const queryKeys = {
  users: ["users"] as const,
  searchUsers: (query: UserSearchQuery) =>
    ["users", "searchUsers", query] as const,
} as const;

export function useSearchUsers(query: UserSearchQuery) {
  const { data: session } = useCurrentSession();

  const role = session?.state === "authenticated" ? session.role : "anonymous";

  return useQuery({
    queryKey: queryKeys.searchUsers(query),
    queryFn: () => listUsers(query),
    enabled: role === "admin",
  });
}

export function useAddUser(onSuccess?: (user: User) => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (user: Omit<User, "id">) => addUser(user),
    onSuccess: (user) => {
      toast.success(`User ${user.email} added`);
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
      onSuccess?.(user);
    },
  });
}

export function useDeleteUser(onSuccess?: () => void) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (userId: string) => deleteUser(userId),
    onSuccess: () => {
      toast.success("User deleted");
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
      onSuccess?.();
    },
  });
}

export function useEditUser(onSuccess?: (user: User) => void) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ userId, user }: UpdateUserOptions) =>
      updateUser({ userId, user }),
    onSuccess: (user) => {
      toast.success(`User ${user.email} updated`);
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
      onSuccess?.(user);
    },
  });
}
