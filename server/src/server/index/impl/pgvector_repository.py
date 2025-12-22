from pathlib import Path
from typing import Any

from sqlmodel import Session, col, func, select

from server.images.models import ImageSearchQuery
from server.index.models import Embedding, IndexedImage
from server.logger import app_logger

_COL_QUERY_MAP: dict[str, Any] = {  # The typing for models is a bit wonky
    "created_at": IndexedImage.created_at,
    "modified_at": IndexedImage.modified_at,
    "user_visible_name": IndexedImage.user_visible_name,
}


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
        try:
            self._db_session.commit()
        except Exception as ex:
            app_logger.error(f"Failed to add images: {ex}")
            self._db_session.rollback()
            raise
        else:
            app_logger.info(f"Added {len(images)} images to the repository.")

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
        embedding: Embedding,
        k: int,
    ) -> list[IndexedImage]:
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

        return list(self._db_session.exec(query).all())

    def get_image_by_path(self, image_path: Path) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its path."""
        statement = select(IndexedImage).where(IndexedImage.path == str(image_path))
        return self._db_session.exec(statement).first()

    def get_image_by_id(self, image_id: int) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its ID."""
        statement = select(IndexedImage).where(IndexedImage.id == image_id)
        return self._db_session.exec(statement).first()

    def query_images(self, query: ImageSearchQuery) -> list[IndexedImage]:
        conditions = list[bool]()  # not really bool, but sqlmodel expression
        if query.name_contains:
            conditions.append(
                IndexedImage.user_visible_name.ilike(
                    f"%{query.name_contains}%",
                ),
            )
        if query.created_min:
            conditions.append(IndexedImage.created_at >= query.created_min)
        if query.created_max:
            conditions.append(IndexedImage.created_at <= query.created_max)
        if query.modified_min:
            conditions.append(IndexedImage.modified_at >= query.modified_min)
        if query.modified_max:
            conditions.append(IndexedImage.modified_at <= query.modified_max)

        if query.direction == "descending":
            ordering = _COL_QUERY_MAP[query.order_by].desc()
        else:
            ordering = _COL_QUERY_MAP[query.order_by].asc()
        stmt = (
            select(IndexedImage)
            .where(*conditions)
            .order_by(ordering)
            .offset((query.page - 1) * query.items_per_page)
            .limit(query.items_per_page)
        )
        app_logger.info(f"ordering by {query.order_by} {query.direction}")
        return list(self._db_session.exec(stmt).all())

    def get_total_images_count(self) -> int:
        """Get the total number of indexed images in the repository."""
        statement = select(func.count(col(IndexedImage.id)))
        return self._db_session.exec(statement).one()
