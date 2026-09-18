"""Run the venue feed in a background thread.

For the Streamlit front end, which has no event loop of its own: its
script is re-executed top to bottom on every rerun, so the feed has to
live somewhere that outlives the script. The websocket server needs none
of this - it is asyncio all the way down already.

The thread publishes finished snapshot dicts under a lock. Readers never
touch the book directly, because `top()` and `depth()` are separate calls
and a read interleaved with an update can show a bid above the ask.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque

from feed import HZ, Feed

HISTORY = 600

# Survives reruns because module globals do. `st.cache_resource` can be
# cleared - by a hot reload during development, by the Clear cache
# button, by eviction - and clearing it does not stop the thread the old
# entry started. Without this, every clear leaves another live websocket
# to the venue behind, and an afternoon of editing ends in a rate limit.
_ACTIVE: list[FeedThread] = []


class FeedThread:
    """Owns the asyncio loop, the venue connection and the latest snapshot.

    One instance per server process, shared by every session.
    """

    def __init__(self) -> None:
        for previous in _ACTIVE:
            previous.stop()
        _ACTIVE.clear()
        _ACTIVE.append(self)

        self.feed = Feed()
        self.mids: deque[float] = deque(maxlen=HISTORY)
        self._lock = threading.Lock()
        self._snapshot: dict | None = None
        self._stopping = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, name="lob-feed", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        client = asyncio.create_task(self.feed.client.run())
        interval = 1.0 / HZ
        try:
            while not self._stopping.is_set():
                await asyncio.sleep(interval)
                self.feed.tick()
                snap = self.feed.snapshot()
                if snap["mid"] is not None:
                    self.mids.append(snap["mid"])
                # Publish the finished dict; readers never touch the book.
                with self._lock:
                    self._snapshot = snap
        finally:
            client.cancel()

    def stop(self) -> None:
        """Wind the feed down from another thread.

        `MarketDataClient.stop` sets an asyncio.Event, which is not safe
        to touch from outside its loop, so it is hopped onto that loop.
        """
        self._stopping.set()
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self.feed.client.stop)

    def latest(self) -> tuple[dict | None, list[float]]:
        with self._lock:
            return self._snapshot, list(self.mids)
