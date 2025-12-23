import { Gallery } from "@/gallery/Gallery";
import { rootRoute } from "@/router/rootRoute";
import { createRoute } from "@tanstack/react-router";

export const galleryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Gallery,
});
