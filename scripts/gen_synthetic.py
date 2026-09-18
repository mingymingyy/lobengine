#!/usr/bin/env python3
"""Generate a synthetic capture so the repo runs with no network.

Writes messages in OKX `books` wire format, which means the synthetic
file is parsed by the real adapter, applied by the real book and checked
by the real gap detection. There is no separate code path for simulated
data.

    python scripts/gen_synthetic.py --events 200000

To exercise gap detection and recovery, drop messages and emit periodic
snapshots so the stream has something to recover to:

    python scripts/gen_synthetic.py --events 50000 --drop-every 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine.recorder import Recorder
from lob_engine.sim import LOBSimulator, SimConfig, to_okx_frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--inst-id", default="SIM-USDT")
    ap.add_argument("--out", default="data/synthetic_okx.jsonl.gz")
    ap.add_argument(
        "--drop-every",
        type=int,
        default=0,
        help="drop every Nth message, to exercise gap detection (0 = off)",
    )
    ap.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help=(
            "emit a full snapshot every N events (0 = off). Defaults to "
            "--drop-every when dropping, so a snapshot lands immediately "
            "after each dropped message, the way a live client gets one "
            "when it resubscribes after a gap. Off otherwise."
        ),
    )
    args = ap.parse_args()

    snapshot_every = args.snapshot_every
    if snapshot_every is None:
        snapshot_every = args.drop_every or 0

    out = Path(args.out)
    if out.exists():
        out.unlink()

    cfg = SimConfig(seed=args.seed)
    events = list(LOBSimulator(cfg).run(args.events))

    # Render from a fresh simulator with the same config, so the opening
    # snapshot reflects the book as it was before the first event.
    frames = to_okx_frames(
        LOBSimulator(cfg),
        events,
        inst_id=args.inst_id,
        snapshot_every=snapshot_every,
        # Two events after the drop: the next delta exposes the gap, then
        # the snapshot repairs it. That is the sequence a live client goes
        # through when it resubscribes.
        snapshot_offset=2 if args.drop_every else 0,
    )

    t0 = time.time_ns()
    written = dropped = snapshots = 0
    update_idx = 0  # counts deltas only, so it lines up with the event index
    with Recorder(out, venue="okx-simulated", symbol=args.inst_id) as rec:
        for t_ns, frame in frames:
            is_snapshot = '"action":"snapshot"' in frame
            if is_snapshot:
                snapshots += 1
            else:
                update_idx += 1
            # Never drop a snapshot: it is what recovery depends on.
            if (
                args.drop_every
                and not is_snapshot
                and update_idx % args.drop_every == 0
            ):
                dropped += 1
                continue
            rec.record(frame, t0 + t_ns)
            written += 1

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {written:,} messages to {out} ({size_mb:.2f} MB)")
    print(f"  snapshots: {snapshots:,}")
    if dropped:
        print(f"  deliberately dropped {dropped:,} messages to exercise gap recovery")
    print(f"simulated span: {events[-1].t_ns / 1e9:,.1f} s of market time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
