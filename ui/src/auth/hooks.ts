import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";
import { sessionQuery, startOtp, verifyOtp } from "./api";
import type { StartOtpInput, VerifyOtpInput } from "./schema";

export function useStartOtp() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  return useMutation({
    mutationFn: (input: StartOtpInput) => startOtp(input),
    onSuccess: (data) => {
      toast.success("OTP sent successfully");
      queryClient.setQueryData(["session"], data);
      navigate({ to: "/login/verify" });
    },
  });
}

export function useVerifyOtp() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  return useMutation({
    mutationFn: (input: VerifyOtpInput) => verifyOtp(input),
    onSuccess: (data) => {
      toast.success("OTP verified successfully");
      queryClient.setQueryData(["session"], data);
      navigate({ to: "/" });
    },
  });
}

export function useCurrentSession() {
  return useQuery(sessionQuery);
}
