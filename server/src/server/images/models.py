from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ImageSearchOrder(BaseModel):
    by: Literal["created_at", "modified_at", "name"] = "modified_at"
    direction: Literal["ascending", "descending"] = "descending"


class ImageSearchFilters(BaseModel):
    name_contains: str | None = None
    created_min: datetime | None = None
    created_max: datetime | None = None
    modified_min: datetime | None = None
    modified_max: datetime | None = None


class ImageSearchQuery(BaseModel):
    page: int = 1
    items_per_page: int = 10
    order: ImageSearchOrder = ImageSearchOrder()
    filters: ImageSearchFilters = ImageSearchFilters()
