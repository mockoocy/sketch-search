import smtplib
import ssl
from email.message import EmailMessage

from pydantic import EmailStr

from server.config.models import SmtpConfig


class SmtpOtpSender:
    def __init__(self, config: SmtpConfig) -> None:
        self._config = config

    def send_otp(self, email: EmailStr, code: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = "Your OTP Code"
        msg["From"] = self._config.from_address
        msg["To"] = email
        msg.set_content(f"Your OTP code is: {code}")
        context = ssl.create_default_context()

        with smtplib.SMTP(
            host=self._config.host,
            port=self._config.port,
        ) as session:
            if self._config.use_tls:
                session.starttls(context=context)
            if self._config.username and self._config.password:
                session.login(self._config.username, self._config.password)
            session.send_message(msg)
