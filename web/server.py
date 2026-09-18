"""Live order-book dashboard, served over a websocket.

The feed itself lives in `web/feed.py`; this module is the ASGI half.

The only design decision here that matters is the one about rates. A
venue's incremental book channel publishes thousands of updates a second;
a browser can paint sixty frames a second and a human can read about
three. So the feed task and the broadcast task are deliberately
decoupled:

    feed task       applies every update to the book, at feed rate.
                    Touches nothing but the book and the OFI accumulator.
    broadcast task  wakes on a fixed timer, reads whatever the book says
                    *now*, serialises once, and sends that one payload to
                    every connected client.

Nothing queues up per-message, so a burst of activity makes the numbers
move faster, never makes the page fall behind. Each client's outbox holds
one payload: if a client is slow, its pending snapshot is replaced rather
than allowed to build a backlog, because a stale book is worth nothing.

Run it with:

    pip install -r web/requirements.txt
    python web/server.py

Configuration is in `web/feed.py`, plus PORT (default 8000) and
MAX_CLIENTS (default 200).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from feed import HZ, VENUE, Feed  # noqa: F401

STATIC = Path(__file__).resolve().parent / "static"

PORT = int(os.environ.get("PORT", "8000"))
MAX_CLIENTS = int(os.environ.get("MAX_CLIENTS", "200"))


class Hub:
    """Fan-out to browsers, latest-snapshot-wins per client."""

    def __init__(self) -> None:
        self.clients: set[asyncio.Queue] = set()

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.clients.add(q)
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        self.clients.discard(q)

    def broadcast(self, payload: str) -> None:
        for q in self.clients:
            if q.full():
                # Drop the snapshot this client never got to; it is stale
                # by definition and the one replacing it is strictly better.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


feed = Feed()
hub = Hub()


async def broadcaster() -> None:
    interval = 1.0 / HZ
    while True:
        await asyncio.sleep(interval)
        # Tick even with nobody watching. Skipping it would let the OFI
        # accumulator run unbounded while the page is empty, and the
        # first person to connect would be shown one enormous window.
        feed.tick()
        if not hub.clients:
            continue  # nobody watching: do not even serialise
        hub.broadcast(json.dumps(feed.snapshot()))


# ---------------------------------------------------------------------- #
# ASGI
# ---------------------------------------------------------------------- #
def _read(name: str) -> bytes:
    return (STATIC / name).read_bytes()


async def http(scope, receive, send) -> None:
    path = scope["path"]
    if path in ("/", "/index.html"):
        body, ctype = _read("index.html"), b"text/html; charset=utf-8"
    elif path == "/healthz":
        body, ctype = b"ok", b"text/plain"
    elif path == "/api/snapshot":
        body = json.dumps(feed.snapshot()).encode()
        ctype = b"application/json"
    else:
        body, ctype = b"not found", b"text/plain"

    status = 404 if body == b"not found" else 200
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", ctype),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def websocket(scope, receive, send) -> None:
    msg = await receive()
    if msg["type"] != "websocket.connect":
        return
    if scope["path"] != "/ws":
        await send({"type": "websocket.close", "code": 1008})
        return
    # A public URL means an unbounded number of viewers. The payload is
    # serialised once and shared, so each extra client costs only one
    # send, but the cap keeps a scraper or a stuck reconnect loop from
    # exhausting the machine's sockets.
    if len(hub.clients) >= MAX_CLIENTS:
        await send({"type": "websocket.close", "code": 1013})  # try again later
        return

    await send({"type": "websocket.accept"})

    q = hub.register()
    # Send one immediately so the page paints without waiting for a tick.
    await send({"type": "websocket.send", "text": json.dumps(feed.snapshot())})

    async def pump() -> None:
        while True:
            await send({"type": "websocket.send", "text": await q.get()})

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            event = await receive()
            if event["type"] == "websocket.disconnect":
                return
    finally:
        pump_task.cancel()
        hub.unregister(q)


async def lifespan(scope, receive, send) -> None:
    tasks: list[asyncio.Task] = []
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            tasks.append(asyncio.create_task(feed.client.run()))
            tasks.append(asyncio.create_task(broadcaster()))
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            feed.client.stop()
            for t in tasks:
                t.cancel()
            await send({"type": "lifespan.shutdown.complete"})
            return


async def app(scope, receive, send) -> None:
    if scope["type"] == "lifespan":
        await lifespan(scope, receive, send)
    elif scope["type"] == "websocket":
        await websocket(scope, receive, send)
    else:
        await http(scope, receive, send)


if __name__ == "__main__":
    import uvicorn

    print(f"[web] {VENUE} {feed.adapter.symbol} -> http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
