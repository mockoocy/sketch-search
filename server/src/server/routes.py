from fastapi import APIRouter
from fastapi.responses import JSONResponse

health_router = APIRouter(prefix="/api")


@health_router.get("/health", tags=["Health"])
def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=200)
