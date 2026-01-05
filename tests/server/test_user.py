import pytest
from fastapi.testclient import TestClient

from tests.conftest import INTEGRATION_TEST_THE_ONLY_USER


@pytest.fixture
def admin_client(default_client: TestClient) -> TestClient:
    response = default_client.post(
        "/api/auth/otp/start",
        json={"email": INTEGRATION_TEST_THE_ONLY_USER},
    )
    assert response.status_code == 200

    challenge_token = default_client.cookies.get("challenge_token")
    assert challenge_token is not None

    code = default_client.app.state.otp_sender.sent[INTEGRATION_TEST_THE_ONLY_USER]
    default_client.cookies.set("challenge_token", challenge_token)
    response = default_client.post(
        "/api/auth/otp/verify",
        json={"code": code},
    )
    assert response.status_code == 200
    assert response.json()["role"] == "admin"
    assert default_client.cookies.get("session_token") is not None
    default_client.cookies.set(
        "session_token",
        default_client.cookies.get("session_token"),
    )
    return default_client


def test_create_user_who_can_login(
    admin_client: TestClient,
) -> None:
    user_data = {
        "email": "new@user.com",
        "role": "user",
    }
    add_user_response = admin_client.post("/api/users/", json=user_data)
    assert add_user_response.status_code == 201
    with TestClient(admin_client.app) as another_client:
        start_otp_response = another_client.post(
            "/api/auth/otp/start",
            json={"email": user_data["email"]},
        )
        assert start_otp_response.status_code == 200
        challenge_token = another_client.cookies.get("challenge_token")
        assert challenge_token is not None
        code = another_client.app.state.otp_sender.sent[user_data["email"]]
        another_client.cookies.set("challenge_token", challenge_token)
        verify_response = another_client.post(
            "/api/auth/otp/verify",
            json={"code": code},
        )
        assert verify_response.status_code == 200
        assert another_client.cookies.get("session_token") is not None


def test_get_user(
    admin_client: TestClient,
) -> None:
    new_user_data = {
        "email": "get@user.com",
        "role": "user",
    }
    add_user_response = admin_client.post("/api/users/", json=new_user_data)
    assert add_user_response.status_code == 201
    user_id = add_user_response.json()["id"]
    get_user_response = admin_client.get(f"/api/users/{user_id}")
    assert get_user_response.status_code == 200
    assert get_user_response.json()["email"] == new_user_data["email"]
    assert get_user_response.json()["role"] == new_user_data["role"]
    assert get_user_response.json()["id"] == user_id


def test_cant_create_same_user_twice(
    admin_client: TestClient,
) -> None:
    user_data = {
        "email": "hello@world.com",
        "role": "user",
    }
    first_response = admin_client.post("/api/users/", json=user_data)
    assert first_response.status_code == 201
    second_response = admin_client.post("/api/users/", json=user_data)
    assert second_response.status_code == 400


def test_delete_user(
    admin_client: TestClient,
) -> None:
    new_user_data = {
        "email": "to@delete.com",
        "role": "user",
    }
    add_user_response = admin_client.post("/api/users/", json=new_user_data)
    assert add_user_response.status_code == 201
    user_id = add_user_response.json()["id"]
    assert add_user_response.json()["email"] == new_user_data["email"]
    assert add_user_response.json()["role"] == new_user_data["role"]
    delete_response = admin_client.delete(f"/api/users/{user_id}")
    assert delete_response.status_code == 204
    get_response = admin_client.get(f"/api/users/{user_id}")
    assert get_response.status_code == 404


def test_update_user(
    admin_client: TestClient,
) -> None:
    new_user_data = {
        "email": "to@update.com",
        "role": "user",
    }
    add_user_response = admin_client.post("/api/users/", json=new_user_data)
    assert add_user_response.status_code == 201
    user_id = add_user_response.json()["id"]
    updated_user_data = {
        "email": "to@update.com",
        "role": "admin",
    }
    update_response = admin_client.put(f"/api/users/{user_id}", json=updated_user_data)
    assert update_response.status_code == 201
    assert update_response.json()["id"] == user_id
    assert update_response.json()["email"] == updated_user_data["email"]
    assert update_response.json()["role"] == updated_user_data["role"]


def test_conflict_on_update_user_email(
    admin_client: TestClient,
) -> None:
    first_user_data = {
        "email": "tobe@conf.licted",
        "role": "user",
    }
    add_first_response = admin_client.post("/api/users/", json=first_user_data)
    assert add_first_response.status_code == 201
    second_user_data = {
        "email": "tobe2@conflicted",
        "role": "user",
    }
    add_second_response = admin_client.post("/api/users/", json=second_user_data)
    assert add_second_response.status_code == 201
    second_user_id = add_second_response.json()["id"]
    conflict_update_data = {
        "email": first_user_data["email"],
        "role": "user",
    }
    update_response = admin_client.put(
        f"/api/users/{second_user_id}",
        json=conflict_update_data,
    )
    assert update_response.status_code == 409


def test_cant_delete_nonexistent_user(
    admin_client: TestClient,
) -> None:
    delete_response = admin_client.delete(
        "/api/users/00000000-0000-0000-0000-000000000000",
    )
    assert delete_response.status_code == 204


@pytest.mark.parametrize("role", ["user", "editor"])
def test_normal_user_cant_access_user_management(
    admin_client: TestClient,
    role: str,
) -> None:
    user_user_data = {
        "email": f"user@{role}.com",
        "role": role,
    }
    add_user_response = admin_client.post("/api/users/", json=user_user_data)
    assert add_user_response.status_code == 201
    with TestClient(admin_client.app) as user_client:
        start_otp_response = user_client.post(
            "/api/auth/otp/start",
            json={"email": user_user_data["email"]},
        )
        assert start_otp_response.status_code == 200
        challenge_token = user_client.cookies.get("challenge_token")
        assert challenge_token is not None
        code = user_client.app.state.otp_sender.sent[user_user_data["email"]]
        user_client.cookies.set("challenge_token", challenge_token)
        verify_response = user_client.post(
            "/api/auth/otp/verify",
            json={"code": code},
        )
        assert verify_response.status_code == 200
        assert user_client.cookies.get("session_token") is not None

        get_users_response = user_client.get("/api/users/")
        assert get_users_response.status_code == 401
        add_user_response = user_client.post("/api/users/", json={})
        assert add_user_response.status_code == 401
        delete_user_response = user_client.delete("/api/users/1")
        assert delete_user_response.status_code == 401
        update_user_response = user_client.put("/api/users/1", json={})
        assert update_user_response.status_code == 401
        edit_user_response = user_client.put("/api/users/1", json={})
        assert edit_user_response.status_code == 401

        user_client.cookies.set(
            "session_token",
            user_client.cookies.get("session_token"),
        )
        get_users_response = user_client.get("/api/users/")
        assert get_users_response.status_code == 403
        add_user_response = user_client.post("/api/users/", json={})
        assert add_user_response.status_code == 403
        delete_user_response = user_client.delete("/api/users/1")
        assert delete_user_response.status_code == 403
        update_user_response = user_client.put("/api/users/1", json={})
        assert update_user_response.status_code == 403
        edit_user_response = user_client.put("/api/users/1", json={})
        assert edit_user_response.status_code == 403
