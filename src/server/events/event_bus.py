from collections import defaultdict
from collections.abc import Callable

from server.events.events import Event

type EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self.subscribers = defaultdict[type[Event], list[EventHandler]](list)

    def subscribe(self, event: type[Event], handler: EventHandler) -> None:
        self.subscribers[event].append(handler)

    def publish(self, event: Event) -> None:
        event_type = type(event)
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                handler(event)
