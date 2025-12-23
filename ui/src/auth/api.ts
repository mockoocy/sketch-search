import { apiFetch } from "@/general/api";
import type { StartOtpInput, VerifyOtpInput } from "./schema";

export async function startOtp(input: StartOtpInput): Promise<void> {
  return apiFetch<void>({
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

export async function verifyOtp(input: VerifyOtpInput): Promise<void> {
  return apiFetch<void>({
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
