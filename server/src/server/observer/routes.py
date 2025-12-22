import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from server.events.event_bus import EventBus
from server.events.events import (
    Event,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

events_router = APIRouter(prefix="/api/events", tags=["events"])


@contextmanager
def server_fs_events(
    event_bus: EventBus,
    queue: asyncio.Queue[Event],
) -> Generator[None, None, None]:
    """
    Context manager to handle subscription and unsubscription
    of events related to file system changes.
    It sets up handlers for sending relevant event data to the provided queue.

    Args:
        event_bus: The event bus to subscribe to.
        queue: The queue to send event data to.
    """

    def on_created(event: FileCreatedEvent) -> None:
        queue.put_nowait(event)

    def on_deleted(event: FileDeletedEvent) -> None:
        queue.put_nowait(event)

    def on_modified(event: FileModifiedEvent) -> None:
        queue.put_nowait(event)

    def on_moved(event: FileMovedEvent) -> None:
        queue.put_nowait(event)

    event_bus.subscribe(
        FileCreatedEvent,
        on_created,
    )
    event_bus.subscribe(
        FileDeletedEvent,
        on_deleted,
    )
    event_bus.subscribe(
        FileModifiedEvent,
        on_modified,
    )
    event_bus.subscribe(
        FileMovedEvent,
        on_moved,
    )
    try:
        yield
    finally:
        event_bus.unsubscribe(
            FileCreatedEvent,
            on_created,
        )
        event_bus.unsubscribe(
            FileDeletedEvent,
            on_deleted,
        )
        event_bus.unsubscribe(
            FileModifiedEvent,
            on_modified,
        )
        event_bus.unsubscribe(
            FileMovedEvent,
            on_moved,
        )


async def _event_generator(
    request: Request,
    queue: asyncio.Queue[Event],
) -> AsyncGenerator[str, None]:
    with server_fs_events(request.app.state.event_bus, queue):
        while not (await request.is_disconnected()):
            event = await queue.get()
            message = f"event: {type(event).__name__}\ndata: {event.as_json()}\n\n"
            yield message


@events_router.get("/")
def fs_events(
    request: Request,
) -> StreamingResponse:
    queue = asyncio.Queue[Event]()
    return StreamingResponse(
        _event_generator(request, queue),
        media_type="text/event-stream",
        headers={
            # required for vite dev server
            "X-Accel-Buffering": "no",
        },
    )
