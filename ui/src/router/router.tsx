import { authRoute } from "@/auth/routes";
import { galleryRoute } from "@/gallery/routes";
import { rootRoute } from "@/router/rootRoute";
import { createRouter } from "@tanstack/react-router";

const routeTree = rootRoute.addChildren([authRoute, galleryRoute]);

export const router = createRouter({
  routeTree,
});
