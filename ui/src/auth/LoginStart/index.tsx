import { useStartOtp } from "@/auth/hooks"
import { startOtpSchema, type StartOtpInput } from "@/auth/schema"
import { Button } from "@/general/components/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/general/components/card"
import { Input } from "@/general/components/input"
import { Label } from "@/general/components/label"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"


export function LoginStart() {
  const {
    register,
    handleSubmit,
    formState: { isValid, errors },
  } = useForm<StartOtpInput>({
    resolver: zodResolver(startOtpSchema),
    mode: "onChange",
  })

  const { mutate } = useStartOtp()

  const onSubmit = (data: StartOtpInput) => {
    mutate(data)
  }

  return (
    <Card className="w-1/3">
      <CardHeader>
        <CardTitle>Login</CardTitle>
        <CardDescription>Enter your email to receive an OTP.</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input id="email" type="email" {...register("email")} />
            {errors.email && (
              <p className="text-sm text-destructive">Invalid email address</p>
            )}
          </div>

          <Button type="submit" className="w-full" disabled={!isValid}>
            Send OTP
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
