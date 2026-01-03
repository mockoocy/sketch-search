import { apiFetch } from "@/general/api";
import {
  anonymousSessionSchema,
  authenticatedSessionSchema,
  challengeIssuesSessionSchema,
  sessionSchema,
  type AnonymousSession,
  type AuthenticatedSession,
  type ChallengeIssuesSession,
  type SessionInfo,
  type StartOtpInput,
  type VerifyOtpInput,
} from "./schema";

export async function startOtp(
  input: StartOtpInput,
): Promise<ChallengeIssuesSession> {
  const data = apiFetch<ChallengeIssuesSession>({
    url: "/api/auth/otp/start",
    context: "Start OTP",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
    credentials: "include",
  });
  return challengeIssuesSessionSchema.parse(await data);
}

export async function verifyOtp(
  input: VerifyOtpInput,
): Promise<AuthenticatedSession> {
  const data = apiFetch<AuthenticatedSession>({
    url: "/api/auth/otp/verify",
    context: "Verify OTP",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
    credentials: "include",
  });
  return authenticatedSessionSchema.parse(await data);
}

export async function getSessionInfo(): Promise<SessionInfo> {
  const data = apiFetch<SessionInfo>({
    url: "/api/session",
    context: "Get Session Info",
    method: "GET",
    credentials: "include",
  });
  return sessionSchema.parse(await data);
}

export const sessionQuery = {
  queryKey: ["session"] as const,
  queryFn: getSessionInfo,
};

export async function logout(): Promise<AnonymousSession> {
  const data = apiFetch<AnonymousSession>({
    url: "/api/session/logout/",
    context: "Logout",
    method: "POST",
    credentials: "include",
  });
  return anonymousSessionSchema.parse(await data);
}
