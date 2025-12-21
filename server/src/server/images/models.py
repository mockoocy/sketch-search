from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ImageSearchQuery(BaseModel):
    page: int = 1
    items_per_page: int = 10
    order_by: Literal["created_at", "modified_at", "user_visible_name"] = "modified_at"
    direction: Literal["ascending", "descending"] = "descending"
    name_contains: str | None = None
    created_min: datetime | None = None
    created_max: datetime | None = None
    modified_min: datetime | None = None
    modified_max: datetime | None = None
