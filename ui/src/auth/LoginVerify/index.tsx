import { zodResolver } from "@hookform/resolvers/zod"
import { useController, useForm } from "react-hook-form"

import {
  InputOTP,
  InputOTPGroup,
  InputOTPSlot,
} from "@/general/components/input-otp"

import { useVerifyOtp } from "@/auth/hooks"
import { verifyOtpSchema, type VerifyOtpInput } from "@/auth/schema"
import { Button } from "@/general/components/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/general/components/card"


export function LoginVerify() {
  const {mutate, isPending, errorMessage} = useVerifyOtp()

  const {
    handleSubmit,
    setValue,
    control,
  } = useForm<VerifyOtpInput>({
    resolver: zodResolver(verifyOtpSchema),
    mode: "onChange",
    defaultValues: {
      code: "",
    },
  })

  const {field, fieldState} = useController({
    control,
    name: "code",
  })

  const code = field.value
  const isValid = !fieldState.invalid

  function onSubmit(data: VerifyOtpInput) {
    mutate(data)
  }


  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle className="text-center">Check your email</CardTitle>
        <CardDescription>
          We sent a one-time code to your email. Enter it below to finish logging
          in.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6 flex flex-col items-center">
          <div className="w-full space-y-3 flex space-x-2">
            <InputOTP
              maxLength={8}
              value={code}
              onChange={(value) =>
                setValue("code", value, { shouldValidate: true })
              }
            >
              <InputOTPGroup>
                <InputOTPSlot index={0} />
                <InputOTPSlot index={1} />
                <InputOTPSlot index={2} />
                <InputOTPSlot index={3} />
                <InputOTPSlot index={4} />
                <InputOTPSlot index={5} />
                <InputOTPSlot index={6} />
                <InputOTPSlot index={7} />
              </InputOTPGroup>
            </InputOTP>

            <div className="flex items-center justify-center text-muted-foreground">
              <span>{code.length}/8</span>
            </div>

          </div>
              <p className="text-sm text-destructive">
                {errorMessage}
              </p>

          <Button
            type="submit"
            className="w-full"
            disabled={!isValid || isPending}
          >
            {isPending ? "Verifying..." : "Verify"}
          </Button>
        </form>

        <div className="text-sm text-muted-foreground">
          If you do not see it, check spam. You can also request a new code from
          the previous step.
        </div>
      </CardContent>
    </Card>
  )
}
