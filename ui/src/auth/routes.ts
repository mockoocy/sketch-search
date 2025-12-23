import { LoginStart } from "@/auth/LoginStart";
import { LoginVerify } from "@/auth/LoginVerify";
import { rootRoute } from "@/router/rootRoute";
import { createRoute } from "@tanstack/react-router";

export const loginRootRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/login",
});

const loginStartRoute = createRoute({
  component: LoginStart,
  path: "/start",
  getParentRoute: () => loginRootRoute,
});

const loginVerifyRoute = createRoute({
  component: LoginVerify,
  path: "/verify",
  getParentRoute: () => loginRootRoute,
});

export const authRoute = loginRootRoute.addChildren([
  loginStartRoute,
  loginVerifyRoute,
]);
