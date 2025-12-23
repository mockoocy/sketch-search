import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";
import { startOtp, verifyOtp } from "./api";
import type { StartOtpInput, VerifyOtpInput } from "./schema";

export function useStartOtp() {
  const navigate = useNavigate();

  return useMutation({
    mutationFn: (input: StartOtpInput) => startOtp(input),
    onSuccess: () => {
      toast.success("OTP sent successfully");
      navigate({ to: "/login/verify" });
    },
  });
}

export function useVerifyOtp() {
  return useMutation({
    mutationFn: (input: VerifyOtpInput) => verifyOtp(input),
    onSuccess: () => {
      toast.success("OTP verified successfully");
    },
  });
}
