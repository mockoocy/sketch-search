from fastapi.testclient import TestClient

from server.config.models import NoAuthConfig, ServerConfig
from server.server import create_app
from tests.conftest import INTEGRATION_TEST_THE_ONLY_USER, CapturingOtpSender


def test_correct_session_state(default_client: TestClient) -> None:
    response = default_client.get("/api/session")
    assert response.status_code == 200
    assert response.json()["state"] == "anonymous"
    response = default_client.post(
        "/api/auth/otp/start",
        json={"email": INTEGRATION_TEST_THE_ONLY_USER},
    )
    response = default_client.get(
        "/api/session",
        cookies={"challenge_token": default_client.cookies.get("challenge_token")},
    )
    assert response.status_code == 200
    assert response.json()["state"] == "challenge_issued"
    challenge_token = default_client.cookies.get("challenge_token")
    otp_sender: CapturingOtpSender = default_client.app.state.otp_sender
    otp_code = otp_sender.sent[INTEGRATION_TEST_THE_ONLY_USER]
    default_client.post(
        "/api/auth/otp/verify",
        json={"code": otp_code},
        cookies={"challenge_token": challenge_token},
    )
    response = default_client.get(
        "/api/session",
        cookies={
            "session_token": default_client.cookies.get("session_token"),
            "challenge_token": default_client.cookies.get("challenge_token"),
        },
    )
    assert response.status_code == 200
    assert response.json()["state"] == "authenticated"
    assert response.json()["role"] == "admin"


def test_logout_clears_challenge_token(default_client: TestClient) -> None:
    response = default_client.post(
        "/api/auth/otp/start",
        json={"email": INTEGRATION_TEST_THE_ONLY_USER},
    )
    assert response.status_code == 200
    assert default_client.cookies.get("challenge_token") is not None
    default_client.post("/api/session/logout")
    assert default_client.cookies.get("challenge_token") is None


def test_logout_clears_session_token(default_client: TestClient) -> None:
    response = default_client.post(
        "/api/auth/otp/start",
        json={"email": INTEGRATION_TEST_THE_ONLY_USER},
    )
    assert response.status_code == 200
    challenge_token = default_client.cookies.get("challenge_token")
    otp_sender: CapturingOtpSender = default_client.app.state.otp_sender
    otp_code = otp_sender.sent[INTEGRATION_TEST_THE_ONLY_USER]
    response = default_client.post(
        "/api/auth/otp/verify",
        json={"code": otp_code},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 200
    assert default_client.cookies.get("session_token") is not None
    default_client.post("/api/session/logout")
    assert default_client.cookies.get("session_token") is None


def test_logout_without_tokens(default_client: TestClient) -> None:
    response = default_client.post("/api/session/logout")
    assert response.status_code == 200


def test_no_auth_config_is_alwaus_admin(settings: ServerConfig) -> None:
    new_settings = settings.model_copy(update={"auth": NoAuthConfig()})
    app = create_app(new_settings)
    with TestClient(app) as client:
        response = client.get("/api/session")
        assert response.status_code == 200
        assert response.json()["state"] == "authenticated"
        assert response.json()["role"] == "admin"
