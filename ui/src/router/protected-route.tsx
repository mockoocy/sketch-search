import { sessionQuery } from "@/auth/api";
import { rootRoute } from "@/router/rootRoute";
import { createRoute, Outlet, redirect } from "@tanstack/react-router";

export const protectedRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: "protected",
  beforeLoad: async ({ context }) => {
    // Example auth check; replace with real logic
    const session = await context.queryClient.ensureQueryData(sessionQuery);
    if (session.state !== "authenticated") {
      throw redirect({ to: "/login/start" });
    }
  },
  component: () => (
    <div className="w-vw h-vh">
      <Outlet />
    </div>
  ),
});
