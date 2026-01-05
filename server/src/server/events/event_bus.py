import contextlib
from collections import defaultdict
from collections.abc import Callable
from typing import cast

from server.events.events import Event

type EventHandler[TEvent] = Callable[[TEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers = defaultdict[type[Event], list[EventHandler[Event]]](list)

    def subscribe[TEvent: Event](
        self,
        event_type: type[TEvent],
        handler: EventHandler[TEvent],
    ) -> None:
        self._subscribers[event_type].append(cast("EventHandler[Event]", handler))

    def unsubscribe[TEvent: Event](
        self,
        event_type: type[TEvent],
        handler: EventHandler[TEvent],
    ) -> None:
        if event_type in self._subscribers:
            with contextlib.suppress(ValueError):
                self._subscribers[event_type].remove(
                    cast("EventHandler[Event]", handler),
                )

    def publish(self, event: Event) -> None:
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                handler(event)
