# Architecture

## The pipeline

```
                    ┌──────────────────────────────────────┐
                    │          venue (websocket)           │
                    └──────────────────┬───────────────────┘
                                       │ raw JSON frames
                                       ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  feeds/client.py          MarketDataClient                       │
   │  • reconnect with exponential backoff + jitter                   │
   │  • stamps recv_ts_ns BEFORE parsing                              │
   │  • on SequenceGap: reset book, resubscribe, resnapshot           │
   └────────┬──────────────────────────────────────────┬──────────────┘
            │                                          │ raw frame
            │ BookUpdate                               ▼
            │                              ┌───────────────────────┐
            │                              │  recorder.py          │
            │                              │  gzipped JSON Lines   │
            │                              │  raw bytes + recv ts  │
            │                              └───────────┬───────────┘
            │                                          │
            │                                          │ replay()
            ▼                                          │
   ┌───────────────────────┐                            │
   │  feeds/exchanges.py   │◄───────────────────────────┘
   │  OKX / Bybit adapter  │
   │  parse() is PURE      │
   └────────┬──────────────┘
            │ BookUpdate (venue-neutral)
            ▼
   ┌───────────────────────┐
   │  book.py  OrderBook   │
   │  • L2 reconstruction  │
   │  • sequence gaps      │
   │  • depth truncation   │
   │  • crossed detection  │
   └────────┬──────────────┘
            │ TopOfBook, depth
            ▼
   ┌───────────────────────┐        ┌────────────────────────────────┐
   │  ofi.py               │        │  strategy.py                   │
   │  order flow imbalance │        │  Avellaneda-Stoikov quoting    │
   └────────┬──────────────┘        └───────────────┬────────────────┘
            │                                       │
            └───────────────┬───────────────────────┘
                            │ desired order
                            ▼
                 ┌──────────────────────┐
                 │  risk.py RiskEngine  │  ← kill switch lives here
                 └──────────┬───────────┘
                            │ approved order
                            ▼
                 ┌──────────────────────┐
                 │  oms.py  OMS         │  ← the single source of truth
                 │  state machine       │     for position and orders
                 └──────────┬───────────┘
                            │ NewOrderSingle
                            ▼
        ┌───────────────────────────────────────────┐
        │  session.py  FixSession                   │
        │  seq nums, heartbeats, resend, gap fill   │
        └───────────────────┬───────────────────────┘
                            │ framed FIX 4.4
                            ▼
                 ┌──────────────────────┐        ┌───────────────────┐
                 │  fix.py  codec       │───────►│  venue.py         │
                 │  BodyLength/CheckSum │        │  matching engine  │
                 └──────────────────────┘        │  price-time       │
                                                 │  priority +       │
                        ExecutionReport ◄────────┤  queue position   │
                                                 └───────────────────┘
```

`latency.py` wraps every stage above. `sim.py` generates the flow that
drives the whole thing when there is no live connection.

## Design decisions and why

### Prices are integers, never floats

Every price inside the engine is an integer number of ticks. Feeds hand
over decimal strings; `PriceScale` converts them exactly.

The reason is that a book is a map from price to size. If the key is a
float, then `0.1 + 0.2 != 0.3` means a delta for a level you already hold
can create a second, near-identical key, and from that moment your book
disagrees with the venue's and never recovers. The bug is silent and
shows up hours later as a mysterious crossed book.

`PriceScale.to_ticks` has a string fast path for the common case and
falls back to `Decimal`. The two are checked against each other in a
differential test, because the fast path exists for speed and must not
quietly become a rounding shortcut.

### Adapter `parse()` is a pure function

`parse(raw, recv_ts) -> list[BookUpdate]` touches no sockets and holds no
state beyond a sequence counter. That single constraint buys:

* the whole pipeline runs offline against recorded bytes
* the simulator can emit real OKX frames and exercise the production path
* there is no separate "test mode" that can drift from live behaviour

### A gap means throw the book away

When `OrderBook.apply` detects that an update does not follow the last
one, it raises and leaves the book **untouched**. It does not apply the
update on a best-effort basis.

That is deliberate. Once you have missed a delta, your book is wrong by
an unknown amount, and every quote you derive from it is wrong too. The
only safe action is to discard and resnapshot. A market maker quoting
from a stale book is a market maker getting picked off.

### The risk layer checks projected position, not current position

Checking `position + this_order` is not enough. If you have 95 working
buy orders resting in the book and a limit of 100, another buy for 10 is
not safe just because you are currently flat. `OMS.exposure` returns the
position you would hold if every working order filled, and that is what
the limit is applied to.

### The matching engine models queue position

`MockVenue` tracks, for every resting order, how much size sits ahead of
it at its price level. That size only decreases through actual executions
and through explicit cancels ahead.

This is the difference between a backtest you can believe and one you
cannot. A trade printing at your price does not fill you. It fills the
people who got there first. A simulator that fills you on every print
will show a market-making strategy earning the spread on volume it would
never have touched, and the resulting P&L is fiction.

### Illegal state transitions raise

The FIX 4.4 order state matrix is encoded as an explicit allow-list. A
fill arriving for an order the OMS believes is already cancelled raises
`OrderStateError` rather than updating position.

A crash is recoverable. A wrong position is not: it propagates into every
risk check, every quote and every P&L number downstream.

### Latency is recorded per stage, as percentiles

Means are useless here. A parse step averaging 5 microseconds with a
p99.9 of 45 microseconds is a step that will miss trades, and the mean
hides that completely. Samples are kept raw, warmup is discarded, and
`time.perf_counter_ns` is the only clock used.

### The simulator models flow, not price

`sim.py` implements four Poisson flows (limit orders, market orders,
cancellations) on a discrete price grid, following Cont, Stoikov and
Talreja (2010). The price is never drawn from a distribution; it moves
only when a queue at the touch empties or a better quote arrives.

This matters for honesty. If the generator assumed a relationship between
order flow and price, then finding that relationship in the regression
would prove nothing. Because the generator assumes nothing of the kind,
the regression result in the README is a genuine emergent property of the
mechanics.

## What is deliberately not here

* **Kernel bypass, busy polling, C++ hot paths.** This is Python. It
  establishes the logic and the measurement harness; it is not a
  production low-latency stack, and pretending otherwise would be worse
  than saying so.
* **Venue CRC checksum validation.** OKX publishes a CRC32 over the top
  25 levels. Implementing it requires retaining the venue's exact string
  formatting, which conflicts with converting to ticks on ingest. Left
  out rather than implemented half-correctly.
* **Full-depth queue modelling on replayed live data.** In the simulator
  queue position is known exactly because the venue is ours. Against a
  real capture it is not observable: a book feed cannot tell a market
  order from a cancellation, so you cannot know whether the size ahead of
  you traded or withdrew.
* **Order entry to a real venue.** The FIX stack talks to `MockVenue`
  only. Nothing in this repo authenticates, signs or sends an order
  anywhere.
