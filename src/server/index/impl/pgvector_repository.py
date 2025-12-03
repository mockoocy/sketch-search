from pathlib import Path

from sqlmodel import Sequence, Session, select

from server.index.models import IndexedImage


class PgVectorIndexedImageRepository:
    def __init__(
        self,
        db_session: Session,
    ) -> None:
        self._db_session = db_session

    def add_images(self, images: list[IndexedImage]) -> None:
        """Add multiple image embeddings to the repository."""
        for image in images:
            self._db_session.add(image)
        self._db_session.commit()

    def delete_image_by_path(self, image_path: Path) -> None:
        """Delete an image embedding from the repository by its ID."""
        statement = select(IndexedImage).where(IndexedImage.path == str(image_path))
        image = self._db_session.exec(statement).first()
        self._db_session.delete(image)
        self._db_session.commit()

    def update_image(self, image: IndexedImage) -> None:
        """Update an existing image embedding in the repository."""
        self._db_session.add(image)
        self._db_session.commit()

    def get_k_nearest_images(
        self,
        embedding: list[float],
        k: int,
    ) -> Sequence[IndexedImage]:
        """Retrieve the k-nearest images to the given image embedding."""
        query = (
            select(
                IndexedImage,
            )
            .order_by(
                IndexedImage.embedding.cosine_distance(embedding),
            )
            .limit(k)
        )
        return self._db_session.exec(query).all()

    def get_image_by_path(self, image_path: Path) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its path."""
        statement = select(IndexedImage).where(IndexedImage.path == str(image_path))
        return self._db_session.exec(statement).first()
