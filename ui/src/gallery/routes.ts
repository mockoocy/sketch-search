import { sessionQuery } from "@/auth/api";
import { Gallery } from "@/gallery/Gallery";
import { protectedRoute } from "@/router/protected-route";
import { createRoute, redirect } from "@tanstack/react-router";

export const galleryRoute = createRoute({
  getParentRoute: () => protectedRoute,
  path: "/",
  component: Gallery,
  beforeLoad: async ({ context }) => {
    document.title = "Gallery";
    const session = await context.queryClient.ensureQueryData(sessionQuery);
    if (session.state !== "authenticated") {
      throw redirect({ to: "/login/start" });
    }
  },
});
