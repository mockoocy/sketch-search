import { rootRoute } from "@/router/rootRoute";
import { createRoute, Outlet } from "@tanstack/react-router";

export const publicRoute = createRoute({
  getParentRoute: () => rootRoute,
  id: "public",
  component: Outlet,
});
