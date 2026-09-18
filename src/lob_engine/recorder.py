"""Capture and replay of raw feed messages.

Two rules that matter more than the file format:

1. **Record the raw payload, not your parsed version.** If your parser has
   a bug you want to be able to fix it and re-run. A capture of parsed
   objects bakes the bug in permanently.
2. **Stamp your own receive time.** The exchange timestamp tells you when
   the venue thinks it sent the message. Your receive timestamp tells you
   when you could first have acted on it. The gap between them is the
   thing you are actually trying to shrink, and you cannot measure it
   afterwards.

Format is gzipped JSON Lines: one object per line, streamable, greppable,
and readable by anything. Parquet would be smaller but adds a dependency
and makes append-during-capture awkward.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Iterator
from pathlib import Path


class Recorder:
    """Append-only writer for raw feed messages."""

    def __init__(self, path: str | Path, venue: str, symbol: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The handle is owned by this object and closed in close()/__exit__;
        # a context manager here would close it before any record is written.
        self._fh = gzip.open(self.path, "at", encoding="utf-8")  # noqa: SIM115
        self.count = 0
        self._write(
            {
                "type": "header",
                "venue": venue,
                "symbol": symbol,
                "started_ns": time.time_ns(),
                "format_version": 1,
            }
        )

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def record(self, payload: str, recv_ts_ns: int | None = None) -> None:
        self._write(
            {
                "type": "msg",
                "recv_ns": recv_ts_ns if recv_ts_ns is not None else time.time_ns(),
                "raw": payload,
            }
        )
        self.count += 1

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._write({"type": "footer", "ended_ns": time.time_ns(), "count": self.count})
        self._fh.close()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_capture(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield (recv_ns, raw_payload) from a capture file."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "msg":
                yield obj["recv_ns"], obj["raw"]


def capture_header(path: str | Path) -> dict:
    """Read just the header record of a capture."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                if obj.get("type") == "header":
                    return obj
                break
    return {}
