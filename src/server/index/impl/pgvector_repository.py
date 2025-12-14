from pathlib import Path
from typing import Any

from sqlmodel import Sequence, Session, select

from server.images.models import ImageSearchQuery
from server.index.models import IndexedImage

_COL_QUERY_MAP: dict[str, Any] = {  # The typing for models is a bit wonky
    "created_at": IndexedImage.created_at,
    "modified_at": IndexedImage.modified_at,
    "name": IndexedImage.user_visible_name,
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

    def get_image_by_id(self, image_id: int) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its ID."""
        statement = select(IndexedImage).where(IndexedImage.id == image_id)
        return self._db_session.exec(statement).first()

    def query_images(self, query: ImageSearchQuery) -> list[IndexedImage]:
        conditions = list[bool]()  # not really bool, but sqlmodel expression
        if query.filters.name_contains:
            conditions.append(
                IndexedImage.user_visible_name.ilike(
                    f"%{query.filters.name_contains}%",
                ),
            )
        if query.filters.created_min:
            conditions.append(IndexedImage.created_at >= query.filters.created_min)
        if query.filters.created_max:
            conditions.append(IndexedImage.created_at <= query.filters.created_max)
        if query.filters.modified_min:
            conditions.append(IndexedImage.modified_at >= query.filters.modified_min)
        if query.filters.modified_max:
            conditions.append(IndexedImage.modified_at <= query.filters.modified_max)

        if query.order.direction == "descending":
            ordering = _COL_QUERY_MAP[query.order.by].desc()
        else:
            ordering = _COL_QUERY_MAP[query.order.by].asc()
        stmt = (
            select(IndexedImage)
            .where(*conditions)
            .order_by(ordering)
            .offset((query.page - 1) * query.items_per_page)
            .limit(query.items_per_page)
        )
        return self._db_session.exec(stmt).all()
