// src/auth/components/RequireRole.tsx
import type { UserRole } from "@/admin/schema";
import { useCurrentSession } from "@/auth/hooks";
import type { ReactNode } from "react";

type RequireRoleProps = {
  role: UserRole;
  children: ReactNode;
};

export function RequireRole({ role, children }: RequireRoleProps) {
  const { data: session } = useCurrentSession();

  if (session?.state !== "authenticated") return null;
  const userRole = session.role;

  if (userRole.indexOf(userRole) < userRole.indexOf(role)) {
    return null;
  }
  return <>{children}</>;
}
