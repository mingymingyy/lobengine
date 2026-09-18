# Pushing this to GitHub

```bash
cd lob-engine
git init
git add .
git commit -m "Order book engine, order flow imbalance, and a FIX order stack"
git branch -M main
git remote add origin git@github.com:<your-username>/lob-engine.git
git push -u origin main
```

The CI workflow in `.github/workflows/ci.yml` runs on push and covers
Python 3.10, 3.11 and 3.12: ruff, the 320 tests, and a smoke run of every
script. Get the green badge before you put the link on an application.

## Repository settings worth doing

- **Description**: "Limit order book reconstruction, order flow imbalance
  (Cont-Kukanov-Stoikov), and a FIX 4.4 order stack with pre-trade risk."
- **Topics**: `market-microstructure` `limit-order-book` `fix-protocol`
  `quantitative-finance` `order-flow-imbalance` `market-making`
- Leave the wiki and projects tabs off. A clean repo reads better.
- Add the CI badge to the top of the README once the first run is green:
  `![CI](https://github.com/<user>/lob-engine/actions/workflows/ci.yml/badge.svg)`

## Before you send the link

1. Run `make all` on your own machine and paste **your** latency numbers
   into the README. The figures currently in there are from the machine
   this was built on and yours will differ.
2. Run `scripts/capture.py` against a live venue for five minutes, then
   `scripts/analyse_ofi.py` on that capture. Add the real-data result to
   the README next to the simulated one. That single step moves the
   project from "implemented a paper" to "tested a paper on live data",
   and it is the strongest thing you can add.
3. Read your own `docs/architecture.md` again. Interviewers ask "why did
   you do it that way" and the answers are all in there.

## The line for your CV or cover letter

> Built a market-data and order-entry stack in Python: websocket L2 book
> reconstruction with sequence-gap detection and resync, a replication of
> the Cont-Kukanov-Stoikov order flow imbalance result, and a FIX 4.4
> order management layer with pre-trade risk controls and a matching
> engine that models queue position. Instrumented per-stage tick-to-order
> latency percentiles; 320 tests.
