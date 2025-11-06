import smtplib
import ssl
from datetime import UTC, datetime
from email.message import EmailMessage

from pydantic import EmailStr
from sqlmodel import Session, select

from server.config_model import OTPAuthConfig, SMTPConfig
from server.db.auth_models import OtpCode, plus
from server.providers.auth_provider import (
    AuthProvider,
    BeginOutcome,
    BeginRequest,
    ChallengeOut,
    IssueOut,
    VerifyRequest,
)
from server.utils import gen_code, hash_code, make_token, new_salt_hex


def send_otp(config: SMTPConfig, to_email: EmailStr, code: str) -> None:
    msg = EmailMessage()
    msg["From"] = config.from_address
    msg["To"] = to_email
    msg["Subject"] = "Your login code"
    msg.set_content(f"Your code is: {code}")
    with smtplib.SMTP(config.host, config.port) as session:
        if config.use_tls:
            session.starttls(context=ssl.create_default_context())
        if config.username:
            session.login(config.username, config.password)
        session.send_message(msg)


class OTPProvider(AuthProvider):
    name = "otp"

    def __init__(self, cfg: OTPAuthConfig) -> None:
        self.cfg = cfg

    def _latest_active(self, session: Session, email: str) -> OtpCode | None:
        query = (
            select(OtpCode)
            .where(
                OtpCode.email == email,
                not OtpCode.consumed,
                OtpCode.expires_at > datetime.now(UTC),
            )
            .order_by(OtpCode.id.desc())
            .limit(1)
        )
        return session.exec(query).first()

    def _create(self, s: Session, email: str, code_hash: str, salt: str) -> OtpCode:
        row = OtpCode(
            email=email,
            code_hash=code_hash,
            salt=salt,
            expires_at=plus(self.cfg.expires_in_s),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row

    def _consume(self, s: Session, row: OtpCode) -> None:
        row.consumed = True
        s.add(row)
        s.commit()

    def _inc_attempts(self, s: Session, row: OtpCode) -> int:
        row.attempts += 1
        s.add(row)
        s.commit()
        return row.attempts

    def begin(self, session: Session, req: BeginRequest) -> BeginOutcome:
        if not req.email:
            err_msg = "email_required"
            raise ValueError(err_msg)
        e = req.email.lower().strip()

        code = gen_code(self.cfg.code_length)
        salt = new_salt_hex()
        self._create(session, e, hash_code(code, salt), salt)

        send_otp(self.cfg.smtp, e, code)
        return ChallengeOut()  # client should navigate to /verify

    def verify(self, session: Session, req: VerifyRequest) -> IssueOut:
        email = req.email.lower().strip()
        row = self._latest_active(session, email)
        if not row:
            err_msg = "no_active_code"
            raise ValueError(err_msg)
        if self._inc_attempts(session, row) > self.cfg.max_attempts:
            err_msg = "too_many_attempts"
            raise ValueError(err_msg)
        if hash_code(req.code, row.salt) != row.code_hash:
            err_msg = "invalid_code"
            raise ValueError(err_msg)

        self._consume(session, row)
        token = make_token(email)
        return IssueOut(access_token=token)
