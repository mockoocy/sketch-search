from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.auth.otp.impl.default_service import DefaultOtpAuthService
from server.auth.otp.routes import otp_router
from server.config.models import SessionConfig
from server.session.impl.default_service import DefaultSessionService
from tests.server.mock.auth import (
    DummyOtpRepository,
    DummyOtpSender,
    DummySessionRepository,
    DummyUserRepository,
)

otp_sender = DummyOtpSender()

app = FastAPI()
app.include_router(otp_router)
app.state.session_service = DefaultSessionService(
    session_repository=DummySessionRepository(),
)
app.state.otp_auth_service = DefaultOtpAuthService(
    session_config=SessionConfig(),
    user_repository=DummyUserRepository(),
    otp_repository=DummyOtpRepository(),
    otp_sender=otp_sender,
)

client = TestClient(app)


def test_otp_auth() -> None:
    response = client.post("/api/auth/otp/start", json={"email": "dummy@example.com"})
    assert response.status_code == 200
    assert client.cookies.get("challenge_token") is not None
    challenge_token = client.cookies.get("challenge_token")
    response = client.post(
        "/api/auth/otp/verify",
        json={"code": otp_sender.otp_container["dummy@example.com"]},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 200
    assert client.cookies.get("session_token") is not None
    assert client.cookies.get("challenge_token") is None


def test_dont_send_otp_to_unknown_email() -> None:
    response = client.post("/api/auth/otp/start", json={"email": "idk@whos.that"})
    assert response.status_code == 404


def test_cant_reuse_otp_code() -> None:
    response = client.post("/api/auth/otp/start", json={"email": "dummy@example.com"})
    assert response.status_code == 200
    challenge_token = client.cookies.get("challenge_token")
    otp_code = otp_sender.otp_container["dummy@example.com"]
    response = client.post(
        "/api/auth/otp/verify",
        json={"code": otp_code},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 200
    response = client.post(
        "/api/auth/otp/verify",
        json={"code": otp_code},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 403


def test_invalid_otp_code() -> None:
    response = client.post("/api/auth/otp/start", json={"email": "dummy@example.com"})
    assert response.status_code == 200
    challenge_token = client.cookies.get("challenge_token")
    response = client.post(
        "/api/auth/otp/verify",
        json={"code": "thisisnottherightcode"},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 401


def test_missing_challenge_token() -> None:
    response = client.post(
        "/api/auth/otp/verify",
        json={"code": "123456"},
    )
    assert response.status_code == 403
