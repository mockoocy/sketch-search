from typing import Protocol


class OtpSender(Protocol):
    def send_otp(self, email: str, code: str) -> None: ...
