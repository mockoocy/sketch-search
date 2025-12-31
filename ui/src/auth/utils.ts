export function hasChallengeToken(): boolean {
  return document.cookie
    .split(";")
    .some((c) => c.trim().startsWith("challenge_token="));
}

export function isAuthenticated(): boolean {
  return document.cookie.includes("session_token=");
}
