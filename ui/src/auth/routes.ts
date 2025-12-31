import { sessionQuery } from "@/auth/api";
import { LoginStart } from "@/auth/LoginStart";
import { LoginVerify } from "@/auth/LoginVerify";
import { rootRoute } from "@/router/rootRoute";
import { createRoute, redirect } from "@tanstack/react-router";

export const loginRootRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/login",
});

const loginStartRoute = createRoute({
  component: LoginStart,
  path: "/start",
  getParentRoute: () => loginRootRoute,
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.ensureQueryData(sessionQuery);
    if (session.state === "authenticated") {
      throw redirect({ to: "/" });
    } else if (session.state === "challenge_issued") {
      throw redirect({ to: "/login/verify" });
    }
  },
});

const loginVerifyRoute = createRoute({
  component: LoginVerify,
  path: "/verify",
  getParentRoute: () => loginRootRoute,
  beforeLoad: async ({ context }) => {
    const session = await context.queryClient.ensureQueryData(sessionQuery);
    if (session.state === "authenticated") {
      throw redirect({ to: "/" });
    } else if (session.state === "anonymous") {
      throw redirect({ to: "/login/start" });
    }
  },
});

export const authRoute = loginRootRoute.addChildren([
  loginStartRoute,
  loginVerifyRoute,
]);
