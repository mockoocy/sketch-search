// Moved to a separate file to avoid circular dependency issues
// Because tanstack/react-router requires routes to reference to
// their parent routes.
// If I were to define it in router.tsx - which imports other routes
// it would create a circular dependency.

import { SharedLayout } from "@/router/SharedLayout";
import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext } from "@tanstack/react-router";

interface RouterContext {
  queryClient: QueryClient;
}

export const rootRoute = createRootRouteWithContext<RouterContext>()({
  component: SharedLayout,
});
