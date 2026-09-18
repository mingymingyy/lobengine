# Live dashboard

A browser view of the reconstructed book: ladder, spread, microprice,
order flow imbalance, cumulative depth, and the feed health counters the
engine already tracks (resyncs, reconnects, parse errors, sequence).

Read-only public market data. Nothing here signs a request or sends an
order; the OMS and FIX half of the engine is deliberately not wired in.

## Two front ends, one feed

| | `web/server.py` | `streamlit_app.py` |
| --- | --- | --- |
| Transport | websocket push | polling reruns |
| Update rate | 10 Hz | ~2 Hz |
| Cost per viewer | one shared `send` | a full script rerun |
| Dependencies | uvicorn | streamlit, pandas, altair |
| Free always-on host | no | Community Cloud |

Both drive the same `Feed` in [feed.py](feed.py) — one venue connection,
one book, one OFI accumulator, one snapshot format. Only the delivery
differs. Pick the websocket app if the point is the engineering; pick
Streamlit if the point is a free link that works.

`feed_thread.py` exists only for Streamlit: that script has no event
loop of its own, so the feed needs somewhere to live that outlasts a
rerun.

## Run it

```bash
pip install -r web/requirements.txt
python web/server.py          # http://127.0.0.1:8000
```

Configuration is all environment variables, all optional:

| Variable         | Default    | Notes                                  |
| ---------------- | ---------- | -------------------------------------- |
| `VENUE`          | `okx`      | `okx` or `bybit`                       |
| `SYMBOL`         | `BTC-USDT` | `BTCUSDT` for bybit                    |
| `PRICE_DECIMALS` | `1`        | `2` for bybit BTCUSDT                  |
| `DEPTH`          | `15`       | ladder rows sent to the browser        |
| `HZ`             | `10`       | broadcast rate                         |
| `PORT`           | `8000`     |                                        |

```bash
VENUE=bybit SYMBOL=BTCUSDT PRICE_DECIMALS=2 python web/server.py
```

`GET /api/snapshot` returns the same payload as one websocket frame,
which is the quickest way to check the feed without a browser.

## How it fits together

```
venue ws ──▶ FeedAdapter.parse ──▶ OrderBook.apply ──▶ OFIAccumulator
                                        │                    │
                                        ▼                    ▼
                                  (feed rate, ~10-10k/s)
                                        │
                              broadcaster, every 1/HZ s
                                        │
                                   one JSON payload
                                        │
                        ┌───────────────┼───────────────┐
                     browser         browser         browser
```

Two rates, on purpose. The feed task applies every update; the broadcast
task samples the book on a timer. Nothing accumulates per-message, so a
burst makes the numbers move faster rather than making the page fall
behind. Each client's outbox holds exactly one payload — a slow client
gets its pending snapshot replaced, never a growing backlog, because a
late book is worth nothing.

## Two things that are easy to get wrong

**Do not send every update.** An incremental book channel can publish
thousands of messages a second. Forwarding each one to the browser
builds an unbounded queue, and what the user sees drifts further behind
the market the busier it gets — the exact moment accuracy matters.

**Do not drive rendering from `requestAnimationFrame` alone.** Browsers
suspend rAF in background tabs, minimised windows and occluded windows.
An rAF-driven page freezes there while the socket is still open and the
indicator still reads "live". The page renders from the message instead,
and a one-second watchdog downgrades the badge to "stalled" if the
server goes quiet.

## Clock skew

`venue lag` is the venue's timestamp against local receive time, so it
spans two machines' clocks. On a host whose clock is off it is
meaningless and can even come out negative; the page labels that case
"clock skew" rather than printing a nonsense latency. `feed age` uses
only the local clock and is always trustworthy — that is the one to
watch for "is the feed alive".

## Deploy

It is a stateful long-lived process holding an outbound websocket, so it
needs somewhere that runs a container. Vercel, Netlify and GitHub Pages
cannot host it: they have nothing to keep running between requests.

That single fact drives the whole hosting decision. The thing being paid
for is not traffic, it is a process that stays awake — so anywhere that
sleeps on idle will drop the venue connection and resync the book from
scratch when it wakes.

**Streamlit Community Cloud**, free and the least work: push the repo,
point it at `streamlit_app.py`, done. It installs the root
`requirements.txt`; there is no Dockerfile or CLI involved. Apps sleep
after a stretch with no visitors and wake on the next one, so the same
cold-start caveat applies — you are trading update rate and a bit of
polish for hosting that costs nothing.

**Render**, no CLI needed either: push this repo, then New → Blueprint
and pick it. `render.yaml` and the `Dockerfile` do the rest. The free
plan spins down after ~15 minutes idle, which is fine for a link someone
opens occasionally — the visitor waits out a cold start and the book
resyncs — but the feed is not running in between. `plan: starter` keeps
it awake.

**Fly**, if you want control over where it runs:

```bash
fly launch --no-deploy     # keep the fly.toml already in this repo
fly deploy
```

`auto_stop_machines = false` and `min_machines_running = 1` are the
settings that matter, for the same reason.

Region is worth a thought: both configs default to Singapore, which is
close to OKX and Bybit. Hosting in, say, Virginia adds a couple of
hundred milliseconds of venue lag to every update, and the `venue lag`
readout will show it.

## Exposing it publicly

Two things in the server exist only because the URL is public.

`/api/snapshot` is a **pure read**. Draining the OFI accumulator is
destructive, so it happens in `Feed.tick()` on the broadcast timer and
nowhere else. Were it in `snapshot()`, a single uptime monitor polling
that endpoint would quietly steal the OFI window from the next broadcast
and flatten the bar for everyone watching.

`MAX_CLIENTS` caps concurrent websockets. The payload is serialised once
and shared, so each extra viewer costs one send and the cap is generous,
but it stops a scraper or a stuck reconnect loop from exhausting the
machine's sockets.

Beyond that there is nothing to lock down: no auth, no user data, no
writes, and the feed is public market data the venue publishes
unauthenticated.
