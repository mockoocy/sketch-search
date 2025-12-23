import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Response, UploadFile
from pydantic import BaseModel

from server.dependencies import image_service, indexing_service, server_config
from server.images.models import ImageSearchQuery
from server.index.models import IndexedImage
from server.logger import app_logger


class SimilaritySearchData(BaseModel):
    image: bytes
    top_k: int


class ListImagesResponse(BaseModel):
    images: list[IndexedImage]
    total: int


images_router = APIRouter(
    prefix="/api/images",
    tags=["images"],
)


@images_router.get("/")
async def list_images(
    image_service: image_service,
    indexing_service: indexing_service,
    query: Annotated[ImageSearchQuery, Depends()],
) -> ListImagesResponse:
    return ListImagesResponse(
        images=image_service.query_images(query),
        total=indexing_service.get_collection_size(),
    )


@images_router.get("/similarity-search/")
async def similarity_search(
    data: Annotated[SimilaritySearchData, Depends()],
    image_service: image_service,
) -> dict[str, list[IndexedImage]]:
    return {
        "images": image_service.similarity_search(data.image, data.top_k),
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
    full_path = (config.watcher.watched_directory / relative_path).resolve()
    indexing_service.embed_images([full_path])
    return {"status": "success"}


@images_router.delete("/{image_id}/")
async def delete_image(
    image_id: int,
    image_service: image_service,
) -> dict[str, str]:
    image_service.remove_image(image_id)
    return {"status": "success"}


@images_router.get("/{image_id}/thumbnail/")
async def get_image_thumbnail(
    image_id: int,
    image_service: image_service,
) -> Response:
    thumbnail = image_service.get_thumbnail_for_image(image_id)
    image_format = thumbnail.format if thumbnail.format else "JPEG"
    app_logger.info(f"Serving thumbnail for image {image_id} in format {image_format}")
    with io.BytesIO() as output:
        thumbnail.save(output, format=image_format)
        return Response(
            content=output.getvalue(),
            media_type=f"image/{image_format.lower()}",
        )
