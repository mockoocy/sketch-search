from fastapi.testclient import TestClient

from tests.conftest import DEFAULT_TEST_USER, CapturingOtpSender


def test_otp_auth(default_client: TestClient) -> None:
    response = default_client.post(
        "/api/auth/otp/start",
        json={"email": DEFAULT_TEST_USER},
    )
    assert response.status_code == 200
    assert response.json()["state"] == "challenge_issued"
    assert default_client.cookies.get("challenge_token") is not None
    challenge_token = default_client.cookies.get("challenge_token")
    otp_sender: CapturingOtpSender = default_client.app.state.otp_sender
    response = default_client.post(
        "/api/auth/otp/verify",
        json={"code": otp_sender.sent[DEFAULT_TEST_USER]},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 200
    assert default_client.cookies.get("session_token") is not None
    assert default_client.cookies.get("challenge_token") is None
    assert response.json()["state"] == "authenticated"
    assert response.json()["role"] == "admin"


def test_dont_send_otp_to_unknown_email(default_client: TestClient) -> None:
    otp_sender: CapturingOtpSender = default_client.app.state.otp_sender
    unknown_email = "un@kno.wn"
    response = default_client.post("/api/auth/otp/start", json={"email": unknown_email})
    assert response.status_code == 200
    assert unknown_email not in otp_sender.sent


def test_cant_reuse_otp_code(default_client: TestClient) -> None:
    otp_sender: CapturingOtpSender = default_client.app.state.otp_sender
    response = default_client.post(
        "/api/auth/otp/start",
        json={"email": DEFAULT_TEST_USER},
    )
    assert response.status_code == 200
    challenge_token = default_client.cookies.get("challenge_token")
    otp_code = otp_sender.sent[DEFAULT_TEST_USER]
    response = default_client.post(
        "/api/auth/otp/verify",
        json={"code": otp_code},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 200
    response = default_client.post(
        "/api/auth/otp/verify",
        json={"code": otp_code},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 403


def test_invalid_otp_code(default_client: TestClient) -> None:
    response = default_client.post(
        "/api/auth/otp/start",
        json={"email": DEFAULT_TEST_USER},
    )
    assert response.status_code == 200
    challenge_token = default_client.cookies.get("challenge_token")
    response = default_client.post(
        "/api/auth/otp/verify",
        json={"code": "thisisnottherightcode"},
        cookies={"challenge_token": challenge_token},
    )
    assert response.status_code == 401


def test_missing_challenge_token(default_client: TestClient) -> None:
    response = default_client.post(
        "/api/auth/otp/verify",
        json={"code": "123456"},
    )
    assert response.status_code == 403
