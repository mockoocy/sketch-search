from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Response, UploadFile

from server.dependencies import image_service, indexing_service, server_config
from server.images.models import ImageSearchQuery
from server.index.models import IndexedImage

DEFAULT_QUERY = ImageSearchQuery()

images_router = APIRouter(
    prefix="/api/images",
    tags=["images"],
)


@images_router.get("/")
async def list_images(
    image_service: image_service,
    query: ImageSearchQuery = DEFAULT_QUERY,
) -> dict[str, list[IndexedImage]]:
    return {
        "images": image_service.query_images(query),
    }


@images_router.get("/similarity-search/")
async def similarity_search(
    image: bytes,
    top_k: int,
    image_service: image_service,
) -> dict[str, list[IndexedImage]]:
    return {
        "images": image_service.similarity_search(image, top_k),
    }


@images_router.get("/search-by-image/")
async def search_by_image(
    image_id: int,
    top_k: int,
    image_service: image_service,
) -> dict[str, list[IndexedImage]]:
    return {
        "images": image_service.search_by_image(image_id, top_k),
    }


@images_router.post("/")
async def add_image(  # noqa: PLR0913
    image: UploadFile,
    image_service: image_service,
    indexing_service: indexing_service,
    response: Response,
    config: server_config,
    directory: Annotated[str, Form(...)],
) -> dict[str, str]:
    if not image.filename:
        response.status_code = 400
        return {"status": "error", "message": "Filename is required."}
    content = await image.read()
    if not content:
        response.status_code = 400
        return {"status": "error", "message": "Empty file."}
    relative_path = Path(directory) / image.filename
    image_service.add_image(content, relative_path)
    full_path = config.watcher.watched_directory / relative_path
    indexing_service.embed_images([full_path])
    return {"status": "success"}


@images_router.delete("/{image_id}/")
async def delete_image(
    image_id: int,
    image_service: image_service,
) -> dict[str, str]:
    image_service.remove_image(image_id)
    return {"status": "success"}
