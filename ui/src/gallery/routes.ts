import { sessionQuery } from "@/auth/api";
import { Gallery } from "@/gallery/Gallery";
import { rootRoute } from "@/router/rootRoute";
import { createRoute, redirect } from "@tanstack/react-router";

export const galleryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Gallery,
  beforeLoad: async ({ context }) => {
    document.title = "Gallery - MyApp";
    const session = await context.queryClient.ensureQueryData(sessionQuery);
    if (session.state !== "authenticated") {
      throw redirect({ to: "/login/start" });
    }
  },
});
