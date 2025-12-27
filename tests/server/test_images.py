from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session

from server.config.models import PostgresConfig, ServerConfig, WatcherConfig
from server.images.impl.default_service import DefaultImageService
from server.images.impl.fs_repository import FsImageRepository
from server.images.routes import images_router
from server.index.impl.default_service import DefaultIndexingService
from server.index.impl.pgvector_repository import PgVectorIndexedImageRepository
from server.index.models import IndexedImage
from tests.server.mock.dummy_embedder import DummyEmbedder


@pytest.fixture
def test_client(tmp_path: Path, db_session: Session) -> Generator[TestClient]:
    app = FastAPI()
    app.include_router(images_router)
    app.state.config = ServerConfig(
        watcher=WatcherConfig(
            watched_directory=str(tmp_path.parent),
        ),
        database=PostgresConfig(
            host="localhost",
            port=5432,
            database="test",
            user="test",
            password="test",  # noqa: S106
        ),
    )

    app.state.image_repository = FsImageRepository()
    app.state.indexed_image_repository = PgVectorIndexedImageRepository(
        db_session=db_session,
    )
    app.state.image_service = DefaultImageService(
        image_repository=app.state.image_repository,
        indexed_image_repository=app.state.indexed_image_repository,
        embedder=DummyEmbedder(),
        thumbnail_config=app.state.config.thumbnail,
        watcher_config=app.state.config.watcher,
    )
    app.state.indexing_service = DefaultIndexingService(
        indexed_repository=app.state.indexed_image_repository,
        embedder=DummyEmbedder(),
    )

    return TestClient(app)


def test_add_image(test_client: TestClient, tmp_path: Path) -> None:
    image_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (32, 32), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    with Path.open(image_path, "wb") as file:
        file.write(buf.getvalue())

    with Path.open(image_path, "rb") as file:
        response = test_client.post(
            "/api/images/",
            files={
                "image": ("test_image.jpg", file, "image/jpeg"),
            },
            data={"directory": "."},
        )
    assert response.status_code == 200


def test_list_images_no_query(test_client: TestClient) -> None:
    response = test_client.get("/api/images/")
    assert response.status_code == 200
    assert "images" in response.json()


def test_list_images_with_query(test_client: TestClient) -> None:
    response = test_client.get(
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


def test_similarity_search(test_client: TestClient) -> None:
    sample_image = bytes([i % 256 for i in range(1536)])
    response = test_client.get(
        "/api/images/similarity-search/",
        params={
            "image": sample_image,
            "top_k": 2,
        },
    )
    assert response.status_code == 200
    assert "images" in response.json()
    assert isinstance(response.json()["images"], list)


def test_search_by_image(test_client: TestClient) -> None:
    test_client.app.state.indexed_image_repository.add_images(
        [
            IndexedImage(
                path="test_search_by_image.jpg",
                user_visible_name="Bobby",
                embedding=np.array([1.0] * 1536),
                content_hash="c",
                model_name="m",
            ),
        ],
    )

    response = test_client.get(
        "/api/images/search-by-image/",
        params={
            "image_id": 1,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    assert "images" in response.json()
    assert isinstance(response.json()["images"], list)


def test_remove_image(test_client: TestClient) -> None:
    """
    Tests if an image with id=1 can be removed.
    Assumes that `test_add_image` has been called before.
    """

    img_path = test_client.app.state.indexed_image_repository.get_image_by_id(1).path
    delete_response = test_client.delete("/api/images/1/")
    assert delete_response.status_code == 200
    assert not Path(img_path).exists()
