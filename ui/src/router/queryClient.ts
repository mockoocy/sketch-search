import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error: Error) => {
      console.log("Query error:", error);
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
      console.log("Mutation error:", error);
    },
  }),
});
