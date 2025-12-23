import { router } from "@/router/router";
import {
  MutationCache,
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { toast } from "sonner";
import "./index.css";

const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error: Error) => {
      toast.error(
        error.message || "An unknown error occurred during a data request.",
      );
    },
  }),
  mutationCache: new MutationCache({
    onError: (error: Error) => {
      toast.error(
        error.message || "An unknown error occurred during a data mutation.",
      );
    },
  }),
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
