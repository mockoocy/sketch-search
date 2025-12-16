from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from sqlmodel import Session

from server.images.models import ImageSearchFilters, ImageSearchOrder, ImageSearchQuery
from server.index.impl.default_service import DefaultIndexingService
from server.index.impl.pgvector_repository import PgVectorIndexedImageRepository
from server.index.models import IndexedImage


def test_embed_images(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    repository = Mock()
    embedder = Mock()
    embedder.name = "model-v1"
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    service = DefaultIndexingService(repository, embedder)

    service.embed_images([img_path])

    images_bytes = embedder.embed.call_args[0][0]
    assert len(images_bytes) == 1
    assert isinstance(images_bytes[0], bytes)

    repository.add_images.assert_called_once()
    added = repository.add_images.call_args[0][0]
    assert len(added) == 1

    img: IndexedImage = added[0]
    assert img.path == str(img_path)
    assert img.embedding == [0.1, 0.2, 0.3]
    assert img.model_name == "model-v1"
    assert img.content_hash is not None
    assert img.created_at is not None
    assert img.modified_at is not None


def test_remove_image() -> None:
    repository = Mock()
    embedder = Mock()
    service = DefaultIndexingService(repository, embedder)

    img_path = Path("/path/to/image.jpg")
    service.remove_image(img_path)

    repository.delete_image_by_path.assert_called_once_with(img_path)


def test_move_image() -> None:
    repository = Mock()
    embedder = Mock()
    service = DefaultIndexingService(repository, embedder)

    old_path = Path("/path/to/old_image.jpg")
    new_path = Path("/path/to/new_image.jpg")

    indexed_image = IndexedImage(
        path=str(old_path),
        user_visible_name="old_image.jpg",
        embedding=([0.1, 0.2, 0.3]),
        created_at=datetime(2005, 4, 2, 21, 37, 0, tzinfo=UTC),
        modified_at=datetime(2025, 4, 2, 21, 37, 0, tzinfo=UTC),
        content_hash="hash",
        model_name="model-v1",
    )
    repository.get_image_by_path.side_effect = [indexed_image, None]

    service.move_image(old_path, new_path)

    repository.get_image_by_path.assert_any_call(old_path)
    repository.get_image_by_path.assert_any_call(new_path)
    repository.update_image.assert_called_once()
    updated_image = repository.update_image.call_args[0][0]
    assert updated_image.path == str(new_path)


def test_get_k_nearest_images(db_session: Session) -> None:
    repo = PgVectorIndexedImageRepository(db_session)

    images = [
        IndexedImage(
            path="a.jpg",
            user_visible_name="a",
            embedding=([1.0] + [0.0] * 1535),
            content_hash="a",
            model_name="m",
        ),
        IndexedImage(
            path="b.jpg",
            user_visible_name="b",
            embedding=([0.0, 1.0] + [0.0] * 1534),
            content_hash="b",
            model_name="m",
        ),
        IndexedImage(
            path="c.jpg",
            user_visible_name="c",
            embedding=([0.9, 0.1] + [0.0] * 1534),
            content_hash="c",
            model_name="m",
        ),
    ]

    repo.add_images(images)

    result = repo.get_k_nearest_images([1.0] + [0.0] * 1535, k=2)

    assert len(result) == 2
    assert result[0].path == "a.jpg"
    assert result[1].path == "c.jpg"


def test_query_images_filters_order_and_pagination(db_session: Session) -> None:
    repo = PgVectorIndexedImageRepository(db_session)

    base_time = datetime(2025, 1, 1, tzinfo=UTC)

    images = [
        IndexedImage(
            path="cat_1.jpg",
            user_visible_name="cat_1",
            embedding=([1.0] + [0.0] * 1535),
            created_at=base_time,
            modified_at=base_time + timedelta(hours=1),
            content_hash="a",
            model_name="m",
        ),
        IndexedImage(
            path="dog_1.jpg",
            user_visible_name="dog_1",
            embedding=([0.0, 1.0] + [0.0] * 1534),
            created_at=base_time + timedelta(days=1),
            modified_at=base_time + timedelta(hours=2),
            content_hash="b",
            model_name="m",
        ),
        IndexedImage(
            path="dog_2.jpg",
            user_visible_name="dog_2",
            embedding=([0.0, 0.0, 1.0] + [0.0] * 1533),
            created_at=base_time + timedelta(days=2),
            modified_at=base_time + timedelta(hours=3),
            content_hash="c",
            model_name="m",
        ),
    ]

    repo.add_images(images)

    query = ImageSearchQuery(
        page=1,
        items_per_page=1,
        filters=ImageSearchFilters(name_contains="dog"),
        order=ImageSearchOrder(by="created_at", direction="ascending"),
    )

    result = repo.query_images(query)

    assert len(result) == 1
    assert result[0].path == "dog_1.jpg"
    # Test second page
    query.page = 2
    result = repo.query_images(query)
    assert len(result) == 1
    assert result[0].path == "dog_2.jpg"
