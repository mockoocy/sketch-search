from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from server.index.impl.default_service import DefaultIndexingService
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
        embedding=[0.1, 0.2, 0.3],
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
