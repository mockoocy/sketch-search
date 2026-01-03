import { z } from "zod";

export const startOtpSchema = z.object({
  email: z.email(),
});

export const verifyOtpSchema = z.object({
  code: z.string().min(8).max(8),
});

export type StartOtpInput = z.infer<typeof startOtpSchema>;
export type VerifyOtpInput = z.infer<typeof verifyOtpSchema>;

export const anonymousSessionSchema = z.object({
  state: z.literal("anonymous"),
});

export const challengeIssuesSessionSchema = z.object({
  state: z.literal("challenge_issued"),
});
export const authenticatedSessionSchema = z.object({
  state: z.literal("authenticated"),
  role: z.enum(["user", "editor", "admin"]),
});

export const sessionSchema = z.discriminatedUnion("state", [
  anonymousSessionSchema,
  challengeIssuesSessionSchema,
  authenticatedSessionSchema,
]);

export type AnonymousSession = z.infer<typeof anonymousSessionSchema>;
export type ChallengeIssuesSession = z.infer<
  typeof challengeIssuesSessionSchema
>;
export type AuthenticatedSession = z.infer<typeof authenticatedSessionSchema>;
export type SessionInfo = z.infer<typeof sessionSchema>;
