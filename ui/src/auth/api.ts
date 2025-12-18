import type { StartOtpInput, VerifyOtpInput } from "./schema"

export class OtpError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "OtpError"
  }
}

export async function startOtp(input: StartOtpInput) {
  const response = await fetch("/api/auth/otp/start", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(input),
      credentials: "include",
    })

  const body =  await response.json()
  if (response.status >= 500) {
    throw new OtpError("Server error occurred")
  }
  if (!response.ok) {
    throw new OtpError(body.error || "Can't start OTP process, unknown error")
  }
  return body
}

export async function verifyOtp(input: VerifyOtpInput) {
  const response = await fetch("/api/auth/otp/verify", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
    credentials: "include",
  })

  const body =  await response.json()
  if (response.status >= 500) {
    throw new OtpError("Server error occurred")
  }
  if (!response.ok) {
    throw new OtpError(body.error || "Can't verify OTP, unknown error")
  }
  return body
}
