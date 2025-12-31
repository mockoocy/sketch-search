import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.auth.guard import auth_guard
from server.dependencies import server_config
from server.events.event_bus import EventBus
from server.events.events import (
    Event,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from server.user.models import UserRole

observer_router = APIRouter(prefix="/api/fs", tags=["observer", "filesystem", "events"])


class DirectoryNode(BaseModel):
    path: str
    parent: str | None
    created_at: datetime
    modified_at: datetime
    children: list["DirectoryNode"] = []


@asynccontextmanager
async def server_fs_events(
    event_bus: EventBus,
    queue: asyncio.Queue[Event],
) -> AsyncGenerator[None, None]:
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
    async with server_fs_events(request.app.state.event_bus, queue):
        while not (await request.is_disconnected()):
            get_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {get_task},
                timeout=15.0,
            )

            if not done:
                get_task.cancel()
                yield ": keep-alive\n\n"
                continue

            event = get_task.result()
            yield f"event: {type(event).__name__}\ndata: {event.as_json()}\n\n"


@observer_router.get("/events/", dependencies=[auth_guard(UserRole.USER)])
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


@observer_router.get("/watched-directories/", dependencies=[auth_guard(UserRole.USER)])
async def watched_directories(
    config: server_config,
) -> Response:
    root = Path(config.watcher.watched_directory)

    def build_tree(path: Path) -> DirectoryNode | None:
        if not path.is_dir():
            return None
        relative_path = path.relative_to(root)
        relative_parent = relative_path.parent
        stat = path.stat()
        node = DirectoryNode(
            path=str(relative_path),
            parent=str(relative_parent) if relative_parent != relative_path else None,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            children=[],
        )
        if path.is_dir():
            for child in path.iterdir():
                child_node = build_tree(child)
                if child_node:
                    node.children.append(child_node)
        return node

    tree = build_tree(root)
    if tree is None:
        return Response(
            status_code=404,
            content="Watched directory not found or is not a directory.",
        )
    return Response(content=tree.model_dump_json(), media_type="application/json")
