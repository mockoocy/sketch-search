from pathlib import Path
from typing import Any
from uuid import UUID

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
        watched_directory: Path,
    ) -> None:
        self._db_session = db_session
        self._watched_directory = watched_directory

    def _build_conditions(self, query: ImageSearchQuery) -> list[bool]:
        conditions = list[bool]()  # not really bool, but sqlmodel expression

        if query.directory == ".":
            conditions.append(~IndexedImage.path.contains("/"))
        elif query.directory:
            prefix = query.directory.rstrip("/")

            conditions.append(
                IndexedImage.path.like(f"{prefix}/%")
                & ~IndexedImage.path.like(f"{prefix}/%/%"),
            )

        if query.name_contains:
            conditions.append(
                IndexedImage.user_visible_name.ilike(f"%{query.name_contains}%"),
            )
        if query.created_min:
            conditions.append(IndexedImage.created_at >= query.created_min)
        if query.created_max:
            conditions.append(IndexedImage.created_at <= query.created_max)
        if query.modified_min:
            conditions.append(IndexedImage.modified_at >= query.modified_min)
        if query.modified_max:
            conditions.append(IndexedImage.modified_at <= query.modified_max)
        return conditions

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
        """Delete an image embedding from the repository by its Path."""
        if image_path.is_relative_to(self._watched_directory):
            relative_path = image_path.relative_to(self._watched_directory)
        else:
            relative_path = image_path

        statement = select(IndexedImage).where(IndexedImage.path == str(relative_path))
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
        search_query: ImageSearchQuery,
    ) -> list[IndexedImage]:
        """Retrieve the k-nearest images to the given image embedding."""
        conditions = self._build_conditions(search_query)

        dist = IndexedImage.embedding.cosine_distance(embedding).label("distance")

        top_stmt = (
            select(IndexedImage.id, dist).where(*conditions).order_by(dist).limit(k)
        ).subquery()

        page_stmt = (
            select(IndexedImage)
            .join(top_stmt, IndexedImage.id == top_stmt.c.id)
            .order_by(top_stmt.c.distance)
            .offset((search_query.page - 1) * search_query.items_per_page)
            .limit(search_query.items_per_page)
        )

        return list(self._db_session.exec(page_stmt).all())

    def get_image_by_path(self, image_path: Path) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its path."""
        statement = select(IndexedImage).where(IndexedImage.path == str(image_path))
        return self._db_session.exec(statement).first()

    def get_image_by_id(self, image_id: UUID) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its ID."""
        statement = select(IndexedImage).where(IndexedImage.id == image_id)
        return self._db_session.exec(statement).first()

    def query_images(self, query: ImageSearchQuery) -> list[IndexedImage]:
        conditions = self._build_conditions(query)
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
