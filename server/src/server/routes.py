from fastapi import APIRouter
from fastapi.responses import JSONResponse

health_router = APIRouter()


@health_router.get("/health", tags=["Health"])
def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=200)
