import { apiFetch } from "@/general/api";
import type { StartOtpInput, VerifyOtpInput } from "./schema";

type UserRole = "user" | "editor" | "admin";

type AnonymousSession = {
  state: "anonymous";
};

type ChallengeIssuesSession = {
  state: "challenge_issued";
};

type AuthenticatedSession = {
  state: "authenticated";
  role: UserRole;
};

export type SessionInfo =
  | AnonymousSession
  | ChallengeIssuesSession
  | AuthenticatedSession;

export async function startOtp(
  input: StartOtpInput,
): Promise<ChallengeIssuesSession> {
  return apiFetch<ChallengeIssuesSession>({
    url: "/api/auth/otp/start",
    context: "Start OTP",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
    credentials: "include",
  });
}

export async function verifyOtp(
  input: VerifyOtpInput,
): Promise<AuthenticatedSession> {
  return apiFetch<AuthenticatedSession>({
    url: "/api/auth/otp/verify",
    context: "Verify OTP",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
    credentials: "include",
  });
}

export async function getSessionInfo(): Promise<SessionInfo> {
  return apiFetch<SessionInfo>({
    url: "/api/auth/session",
    context: "Get Session Info",
    method: "GET",
    credentials: "include",
  });
}

export const sessionQuery = {
  queryKey: ["session"] as const,
  queryFn: getSessionInfo,
};
