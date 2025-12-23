import { z } from "zod";

export const startOtpSchema = z.object({
  email: z.email(),
});

export const verifyOtpSchema = z.object({
  code: z.string().min(8).max(8),
});

export type StartOtpInput = z.infer<typeof startOtpSchema>;
export type VerifyOtpInput = z.infer<typeof verifyOtpSchema>;
