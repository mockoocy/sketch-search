import asyncio
from collections.abc import AsyncGenerator
import logging
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from watchdog.events import FileSystemEvent
from fastapi.responses import StreamingResponse
from server.fs_observer import (
    start_observer,
    shutdown_observer,
    ImageFsEvent,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.observer = start_observer()
    yield
    shutdown_observer(app.state.observer)

APP = FastAPI(lifespan=lifespan)


async def fs_events(request: Request) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[FileSystemEvent] = asyncio.Queue()

    def _handler(event: FileSystemEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    for event in ImageFsEvent:
        event.signal.connect(_handler)

    try:
        while not await request.is_disconnected():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=2.5)
            except TimeoutError:
                # This is here to avoid hanging connections
                continue
            yield str(event)
    except asyncio.TimeoutError:
        logging.getLogger("uvicorn.error").info("Client disconnected from /events")
    finally:
        for event in ImageFsEvent:
            event.signal.disconnect(_handler)



@APP.get("/events")
async def get_events(request: Request)-> StreamingResponse:
    return StreamingResponse(fs_events(request), media_type="text/event-stream")
