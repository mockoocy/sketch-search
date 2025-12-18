import { useMutation } from "@tanstack/react-query"
import { useNavigate } from "@tanstack/react-router"
import { toast } from "sonner"
import { OtpError, startOtp, verifyOtp } from "./api"
import type { StartOtpInput, VerifyOtpInput } from "./schema"

export function useStartOtp() {

  const navigate = useNavigate();


  const mutation = useMutation({
    mutationFn: (input: StartOtpInput) => startOtp(input),
    onError: (error: Error) => {
      console.error(error)
      toast.error(otpErrorMessage(error))
    },
    onSuccess: () => {
      toast.success("OTP sent successfully")
      navigate({ to: "/login/verify" })
    }
  })
  const errorMessage = mutation.error ? otpErrorMessage(mutation.error) : null;
  return { ...mutation, errorMessage };

}

export function useVerifyOtp() {
  const mutation = useMutation({
    mutationFn: (input: VerifyOtpInput) => verifyOtp(input),
    onSuccess: () => {
      toast.success("OTP verified successfully")
    },
    onError: (error: Error) => {
      console.error(error)
      toast.error(otpErrorMessage(error))
    },
  })
  const errorMessage = mutation.error ? otpErrorMessage(mutation.error) : null;
  return { ...mutation, errorMessage };
}


function otpErrorMessage(error: Error): string {
  if (!(error instanceof OtpError)) {
    return 'An unexpected error occurred'
  }

  return error.message
}
