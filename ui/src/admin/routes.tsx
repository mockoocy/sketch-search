import { AdminPage } from "@/admin/AdminPage";
import { sessionQuery } from "@/auth/api";
import { protectedRoute } from "@/router/protected-route";
import { createRoute, redirect } from "@tanstack/react-router";

export const adminRoute = createRoute({
  getParentRoute: () => protectedRoute,
  path: "/admin",
  component: AdminPage,

  beforeLoad: async ({ context }) => {
    document.title = "Admin - Sketch Search";
    const session = await context.queryClient.ensureQueryData(sessionQuery);
    if (!(session.state === "authenticated" && session.role === "admin")) {
      // will trigger further redirects if not authenticated
      throw redirect({ to: "/" });
    }
  },
});
