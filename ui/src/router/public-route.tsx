import { rootRoute } from "@/router/rootRoute";
import { createRoute, Outlet } from "@tanstack/react-router";

export const publicRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: "public",
  component: () => (
    <div className="w-full h-full flex items-center justify-center">
      <Outlet />
    </div>
  ),
});
