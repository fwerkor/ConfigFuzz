import collections
from typing import Deque, Optional

import msadapter


class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: Deque[msadapter.Event] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen

    def enqueue(self, free_event: msadapter.Event) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> Optional[msadapter.Event]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> Optional[msadapter.Event]:
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None
