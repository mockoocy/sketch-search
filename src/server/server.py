import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlmodel import Session
from watchdog.events import FileSystemEvent

from server.config_model import get_server_config
from server.db.db_core import get_session, init_db
from server.fs_observer import (
    ImageFsEvent,
    shutdown_observer,
    start_observer,
)
from server.providers.auth_provider import (
    AuthProvider,
    BeginRequest,
    IssueOut,
    VerifyRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    app.state.observer = start_observer()
    yield

    shutdown_observer(app.state.observer)


APP = FastAPI(lifespan=lifespan)
APP.state.config = get_server_config()


async def fs_events(request: Request) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[FileSystemEvent] = asyncio.Queue()

    def _handler(event: FileSystemEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    for event in ImageFsEvent:
        event.signal.connect(_handler)

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.5)
        except TimeoutError:
            # This is here to avoid hanging connections
            continue
        yield str(event)


@APP.get("/events")
async def get_events(request: Request) -> StreamingResponse:
    return StreamingResponse(fs_events(request), media_type="text/event-stream")


@APP.get("/health")
async def health_check() -> dict[Literal["status"], Literal["ok"]]:
    return {"status": "ok"}


def get_provider() -> AuthProvider:
    server_config = get_server_config()
    if server_config.auth.kind == "otp":
        from server.providers.otp_auth_provider import OTPProvider

        return OTPProvider(server_config.auth)
    from server.providers.no_auth_provider import NoAuthProvider

    return NoAuthProvider()


@APP.post("/start")
async def start_login(
    data: BeginRequest,
    session: Annotated[Session, Depends(get_session)],
    provider: Annotated[AuthProvider, Depends(get_provider)],
) -> RedirectResponse:
    out = provider.begin(session, data)
    if isinstance(out, IssueOut):
        # /health is a dummy endpoint here
        return RedirectResponse("/health", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse("/verify", status_code=status.HTTP_303_SEE_OTHER)


@APP.post("/verify")
async def verify_login(
    data: VerifyRequest,
    session: Annotated[Session, Depends(get_session)],
    provider: Annotated[AuthProvider, Depends(get_provider)],
) -> RedirectResponse:
    provider.verify(session, data)
    return RedirectResponse("/health", status_code=status.HTTP_303_SEE_OTHER)
