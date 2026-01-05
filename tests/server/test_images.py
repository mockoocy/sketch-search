import json
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

if TYPE_CHECKING:
    from server.images.service import ImageService


def test_add_image(no_auth_client: TestClient, tmp_path: Path) -> None:
    image_path = tmp_path / "watched" / "test_image.jpg"
    img = Image.new("RGB", (32, 32), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    with Path.open(image_path, "wb") as file:
        file.write(buf.getvalue())

    with Path.open(image_path, "rb") as file:
        response = no_auth_client.post(
            "/api/images/",
            files={
                "image": ("test_image.jpg", file, "image/jpeg"),
            },
            data={"directory": "."},
        )
    assert response.status_code == 200


def test_list_images_no_query(no_auth_client: TestClient) -> None:
    response = no_auth_client.get("/api/images/")
    assert response.status_code == 200
    assert "images" in response.json()


def test_list_images_with_query(no_auth_client: TestClient) -> None:
    response = no_auth_client.get(
        "/api/images/",
        params={
            "page": 2,
            "items_per_page": 5,
            "order_by": "user_visible_name",
            "direction": "ascending",
            "name_contains": "sample",
        },
    )
    assert response.status_code == 200
    assert "images" in response.json()
    assert isinstance(response.json()["images"], list)


def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_similarity_search(no_auth_client: TestClient, tmp_path: Path) -> None:
    watched_dir = tmp_path / "watched"

    first_image = np.array([list(range(32)) for _ in range(32 * 3)])
    almost_first_image = np.array(
        [[(i + 1) % 256 for i in range(32)] for _ in range(32 * 3)],
    )
    third_image = np.array(
        [[(i + 100) % 256 for i in range(32)] for _ in range(32 * 3)],
    )

    Image.fromarray(np.array(first_image, dtype=np.uint8).reshape((32, 32, 3))).save(
        watched_dir / "first_image.jpg",
    )
    Image.fromarray(
        np.array(almost_first_image, dtype=np.uint8).reshape((32, 32, 3)),
    ).save(
        watched_dir / "almost_first_image.jpg",
    )
    Image.fromarray(np.array(third_image, dtype=np.uint8).reshape((32, 32, 3))).save(
        watched_dir / "third_image.jpg",
    )

    sleep(3)

    query_img = Image.fromarray(
        np.array(first_image, dtype=np.uint8).reshape((32, 32, 3)),
    )
    query_bytes = _pil_to_jpeg_bytes(query_img)

    query = {
        "page": 1,
        "items_per_page": 2,
    }

    response = no_auth_client.post(
        "/api/images/similarity-search/",
        data={
            "top_k": "2",
            "query_json": json.dumps(query),
        },
        files={
            "image": ("query.jpg", query_bytes, "image/jpeg"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "images" in body
    assert len(body["images"]) == 2

    returned_names = {
        img.get("user_visible_name") or img.get("file_name") for img in body["images"]
    }

    assert body["total"] == 2
    assert "first_image.jpg" in returned_names
    assert "almost_first_image.jpg" in returned_names


def test_search_by_image(no_auth_client: TestClient, tmp_path: Path) -> None:
    watched_dir = tmp_path / "watched"
    stupid_image = np.array([list(range(32)) for _ in range(32 * 3)])
    slightly_different_image = np.array(
        [[(i + 1) % 256 for i in range(32)] for _ in range(32 * 3)],
    )
    very_different_image = np.array(
        [[(i + 100) % 256 for i in range(32)] for _ in range(32 * 3)],
    )
    stupid_image_pil = Image.fromarray(
        np.array(stupid_image, dtype=np.uint8).reshape((32, 32, 3)),
    )
    stupid_image_pil.save(watched_dir / "stupid_image.jpg")
    slightly_different_image_pil = Image.fromarray(
        np.array(slightly_different_image, dtype=np.uint8).reshape((32, 32, 3)),
    )
    slightly_different_image_pil.save(watched_dir / "slightly_different_image.jpg")
    very_different_image_pil = Image.fromarray(
        np.array(very_different_image, dtype=np.uint8).reshape((32, 32, 3)),
    )
    very_different_image_pil.save(watched_dir / "very_different_image.jpg")
    sleep(3)  # Wait for the image service to index the new images
    img_service: ImageService = no_auth_client.app.state.image_service
    very_stupid_image_indexed = img_service.get_image_by_path("stupid_image.jpg")
    assert very_stupid_image_indexed is not None
    response = no_auth_client.post(
        "/api/images/search-by-image/",
        json={
            "image_id": str(very_stupid_image_indexed.id),
            "top_k": 2,
            "query": {
                "page": 1,
                "items_per_page": 2,
            },
        },
    )
    assert response.status_code == 200
    assert "images" in response.json()
    returned_images = response.json()["images"]
    assert len(returned_images) == 2
    returned_image_ids = {img["id"] for img in returned_images}
    assert str(very_stupid_image_indexed.id) in returned_image_ids
    slightly_different_image_indexed = img_service.get_image_by_path(
        "slightly_different_image.jpg",
    )
    assert slightly_different_image_indexed is not None
    assert str(slightly_different_image_indexed.id) in returned_image_ids


def test_remove_image(no_auth_client: TestClient, tmp_path: Path) -> None:
    watch_dir = tmp_path / "watched"
    image_path = watch_dir / "test_remove_image.jpg"
    img = Image.new("RGB", (32, 32), color="blue")
    img.save(image_path)

    image_service: ImageService = no_auth_client.app.state.image_service
    sleep(2)
    new_img = image_service.get_image_by_path("test_remove_image.jpg")
    assert new_img is not None

    response = no_auth_client.delete(f"/api/images/{new_img.id}/")
    assert response.status_code == 200
    sleep(2)
    deleted_img = image_service.get_image_by_path("test_remove_image.jpg")
    assert deleted_img is None
