"""Latency measurement.

Averages hide everything that matters. A handler with a 20 microsecond
mean and a 4 millisecond p99.9 is a handler that will miss the trades you
actually wanted. So this records every sample and reports percentiles.

Three rules baked in:

* `time.perf_counter_ns` only. `time.time` is wall clock, is not
  monotonic, and has coarser resolution.
* The recorder pre-allocates and appends to a list; no formatting, no
  dict lookups, no locks in the measured path.
* Warmup samples are discarded, because the first pass through a code
  path in CPython is dominated by import and branch-prediction effects
  that never recur.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LatencyReport:
    name: str
    count: int
    p50_ns: float
    p90_ns: float
    p99_ns: float
    p999_ns: float
    max_ns: float
    mean_ns: float

    def as_row(self) -> str:
        def us(x: float) -> str:
            return f"{x / 1000.0:.2f}"

        return (
            f"{self.name:<28}{self.count:>9}{us(self.p50_ns):>10}"
            f"{us(self.p90_ns):>10}{us(self.p99_ns):>10}"
            f"{us(self.p999_ns):>11}{us(self.max_ns):>10}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'stage':<28}{'count':>9}{'p50':>10}{'p90':>10}"
            f"{'p99':>10}{'p99.9':>11}{'max':>10}\n"
            + "-" * 88
            + "\n(all figures in microseconds)"
        )


class LatencyRecorder:
    """Collects nanosecond samples for one named stage."""

    __slots__ = ("name", "_samples", "_warmup", "_seen")

    def __init__(self, name: str, warmup: int = 0) -> None:
        self.name = name
        self._samples: list[int] = []
        self._warmup = warmup
        self._seen = 0

    def record(self, ns: int) -> None:
        self._seen += 1
        if self._seen > self._warmup:
            self._samples.append(ns)

    @contextmanager
    def time(self):
        """`with rec.time(): ...` measures the block."""
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(time.perf_counter_ns() - t0)

    def reset(self) -> None:
        self._samples.clear()
        self._seen = 0

    @property
    def samples(self) -> list[int]:
        return self._samples

    def report(self) -> LatencyReport:
        if not self._samples:
            nan = float("nan")
            return LatencyReport(self.name, 0, nan, nan, nan, nan, nan, nan)
        a = np.asarray(self._samples, dtype=float)
        p50, p90, p99, p999 = np.percentile(a, [50, 90, 99, 99.9])
        return LatencyReport(
            name=self.name,
            count=int(a.size),
            p50_ns=float(p50),
            p90_ns=float(p90),
            p99_ns=float(p99),
            p999_ns=float(p999),
            max_ns=float(a.max()),
            mean_ns=float(a.mean()),
        )


class LatencyBook:
    """A named collection of recorders."""

    def __init__(self, warmup: int = 0) -> None:
        self._warmup = warmup
        self._recorders: dict[str, LatencyRecorder] = {}

    def __getitem__(self, name: str) -> LatencyRecorder:
        rec = self._recorders.get(name)
        if rec is None:
            rec = LatencyRecorder(name, warmup=self._warmup)
            self._recorders[name] = rec
        return rec

    def reports(self) -> list[LatencyReport]:
        return [r.report() for r in self._recorders.values()]

    def render(self) -> str:
        lines = [LatencyReport.header()]
        lines += [r.as_row() for r in self.reports()]
        return "\n".join(lines)
