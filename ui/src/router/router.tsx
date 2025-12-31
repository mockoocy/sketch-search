import { adminRoute } from "@/admin/routes";
import { authRoute } from "@/auth/routes";
import { galleryRoute } from "@/gallery/routes";
import { protectedRoute } from "@/router/protected-route";
import { publicRoute } from "@/router/public-route";
import { queryClient } from "@/router/queryClient";
import { rootRoute } from "@/router/rootRoute";
import { createRouter } from "@tanstack/react-router";

const routeTree = rootRoute.addChildren([
  protectedRoute.addChildren([adminRoute, galleryRoute]),
  publicRoute.addChildren([authRoute]),
]);

export const router = createRouter({
  routeTree,
  context: {
    queryClient,
  },
});
