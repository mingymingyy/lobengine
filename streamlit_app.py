"""lob-engine live book, as a Streamlit app.

Same engine and the same `Feed` object as the websocket server in
`web/server.py`; only the presentation differs. Streamlit earns its
place here for one reason: Community Cloud hosts it free from a GitHub
repo, with no Dockerfile, no CLI and no card.

One thing about Streamlit's execution model shapes this file. The script
is re-executed top to bottom for every viewer and every refresh, so a
venue connection opened at module scope would mean one websocket per
viewer per rerun - a rate limit within minutes. `st.cache_resource` is
the fix: it is global to the server process, so the feed thread is
created once and every session shares it. The thread itself, and the
locking that keeps a mid-update read from showing a crossed book, live
in `web/feed_thread.py`.

Run locally:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "web"))

from feed import DEPTH  # noqa: E402
from feed_thread import FeedThread  # noqa: E402

BID_GREEN = "#2ea36b"
ASK_RED = "#d9534f"
REFRESH = "0.5s"  # Streamlit reruns the fragment; 2 Hz reads comfortably.


@st.cache_resource
def get_feed() -> FeedThread:
    """One feed per server process, shared across all sessions."""
    return FeedThread()


def fmt(value: float | None, decimals: int) -> str:
    return "—" if value is None else f"{value:,.{decimals}f}"


def cumulative_from_touch(levels: list) -> list[float]:
    """Running total outward from the best price."""
    out, running = [], 0.0
    for _, size in levels:
        running += size
        out.append(running)
    return out


def cumulative_from_far(levels: list) -> list[float]:
    """Running total for rows printed far-touch-first.

    The ask ladder is shown descending toward the spread, so its top row
    is the deepest level and carries the full total.
    """
    return list(reversed(cumulative_from_touch(levels)))


st.set_page_config(
    page_title="lob-engine live book",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

feed_thread = get_feed()

st.title("Live order book")
st.caption(
    "Reconstructed from the venue's incremental feed by "
    "[lob-engine](https://github.com/mingymingyy/lobengine). "
    "Read-only public market data."
)


@st.fragment(run_every=REFRESH)
def live_book() -> None:
    snap, mids = feed_thread.latest()

    if snap is None or not snap["ready"]:
        st.info(
            "Waiting for the first snapshot from the venue…",
            icon=":material/sync:",
        )
        return

    dp = snap["price_decimals"]
    stats = snap["stats"]

    st.subheader(f"{snap['symbol']} · {snap['venue']}", divider="grey")

    with st.container(horizontal=True):
        st.metric(
            "Mid",
            fmt(snap["mid"], dp),
            (
                f"{snap['mid'] - mids[0]:+,.{dp}f} this window"
                if mids and snap["mid"] is not None
                else None
            ),
            border=True,
        )
        st.metric(
            "Spread",
            fmt(snap["spread"], dp),
            f"{snap['spread_ticks']} ticks" if snap["spread_ticks"] else None,
            delta_color="off",
            border=True,
        )
        st.metric(
            "Microprice",
            fmt(snap["microprice"], dp),
            (
                f"{snap['microprice'] - snap['mid']:+.{dp + 2}f} vs mid"
                if snap["microprice"] is not None and snap["mid"] is not None
                else None
            ),
            border=True,
        )
        st.metric(
            "Feed age",
            f"{snap['feed_age_ms']:.0f} ms" if snap["feed_age_ms"] is not None else "—",
            border=True,
        )
        st.metric("Msg rate", f"{stats['msg_rate']:,.0f}/s", border=True)

    # `venue lag` spans two machines' clocks. On an unsynchronised host
    # it is meaningless and can invert; say so rather than print a
    # nonsense latency beside numbers that are real.
    lag = snap["venue_lag_ms"]
    if lag is not None and lag < 0:
        st.caption(
            f":orange[Local clock is {-lag:.0f} ms behind the venue, so venue lag "
            "is not measurable here.] Feed age uses only the local clock."
        )
    elif lag is not None:
        st.caption(f"Venue lag {lag:.0f} ms · feed age uses only the local clock.")

    ladder_col, chart_col = st.columns([1, 1])

    with ladder_col:
        with st.container(border=True):
            st.markdown("**Order book**")

            # Two tables rather than one. ProgressColumn colours a whole
            # column, so a single table would paint the bid bars in the
            # sell colour - backwards, in the one UI where red and green
            # carry meaning. Split by side, and the spread lands where it
            # belongs: between them.
            asks = [
                {"Price": p, "Size": s, "Cumulative": c}
                for (p, s), c in zip(
                    reversed(snap["asks"]),
                    cumulative_from_far(snap["asks"]),
                    strict=True,
                )
            ]
            bids = [
                {"Price": p, "Size": s, "Cumulative": c}
                for (p, s), c in zip(
                    snap["bids"], cumulative_from_touch(snap["bids"]), strict=True
                )
            ]

            sizes = [r["Size"] for r in asks + bids]
            max_size = max(sizes) if sizes else 1.0

            def ladder_config(colour: str) -> dict:
                return {
                    "Price": st.column_config.NumberColumn(format=f"%.{dp}f"),
                    "Size": st.column_config.ProgressColumn(
                        format="compact",
                        min_value=0.0,
                        max_value=max_size,
                        color=colour,
                    ),
                    "Cumulative": st.column_config.NumberColumn(format="compact"),
                }

            st.dataframe(
                pd.DataFrame(asks),
                hide_index=True,
                height=35 * min(len(asks), DEPTH) + 38,
                column_config=ladder_config(ASK_RED),
            )
            st.caption(
                f"**Spread {fmt(snap['spread'], dp)}** · mid {fmt(snap['mid'], dp)}"
            )
            st.dataframe(
                pd.DataFrame(bids),
                hide_index=True,
                height=35 * min(len(bids), DEPTH) + 38,
                column_config=ladder_config(BID_GREEN),
            )

    with chart_col:
        with st.container(border=True):
            st.markdown("**Mid price**")
            if len(mids) > 1:
                # Altair rather than st.line_chart: a price series must
                # not be zero-baselined. Anchored at zero, a book moving
                # 20 bps on a 77,000 mid renders as a dead flat line.
                history = pd.DataFrame({"t": range(len(mids)), "mid": mids})
                st.altair_chart(
                    alt.Chart(history)
                    .mark_line(color=BID_GREEN)
                    .encode(
                        x=alt.X("t:Q", axis=None),
                        y=alt.Y(
                            "mid:Q",
                            scale=alt.Scale(zero=False, nice=False),
                            title=None,
                        ),
                    )
                    .properties(height=200),
                    width="stretch",
                )
            else:
                st.caption("Collecting history…")

        with st.container(border=True):
            st.markdown("**Cumulative depth**")
            # Same reason on the x axis: prices cluster in a narrow band
            # well away from zero. Step interpolation because a book
            # really is a staircase, and each side steps the way it fills.
            bid_rows = [
                {"Price": p, "Depth": c}
                for (p, _), c in zip(
                    snap["bids"], cumulative_from_touch(snap["bids"]), strict=True
                )
            ]
            ask_rows = [
                {"Price": p, "Depth": c}
                for (p, _), c in zip(
                    snap["asks"], cumulative_from_touch(snap["asks"]), strict=True
                )
            ]

            price_axis = alt.X(
                "Price:Q", scale=alt.Scale(zero=False, nice=False), title=None
            )
            depth_axis = alt.Y("Depth:Q", title=None)
            bid_area = (
                alt.Chart(pd.DataFrame(bid_rows))
                .mark_area(color=BID_GREEN, opacity=0.55, interpolate="step-before")
                .encode(x=price_axis, y=depth_axis)
            )
            ask_area = (
                alt.Chart(pd.DataFrame(ask_rows))
                .mark_area(color=ASK_RED, opacity=0.55, interpolate="step-after")
                .encode(x=price_axis, y=depth_axis)
            )
            st.altair_chart(
                (bid_area + ask_area).properties(height=180), width="stretch"
            )

    with st.container(border=True):
        st.markdown("**Order flow imbalance**")
        st.caption(
            "Cont–Kukanov–Stoikov, per tick window. "
            "Positive is net buying pressure."
        )
        st.metric(
            "This window",
            f"{snap['ofi_window']:+.3f}",
            f"{snap['ofi_events']} events",
            delta_color="off",
        )

    # The engine's own counters. They are the reason to trust - or not
    # trust - every number above them.
    with st.container(horizontal=True):
        st.metric("Messages", f"{stats['messages']:,}", border=True)
        st.metric("Updates", f"{stats['updates']:,}", border=True)
        st.metric("Resyncs", stats["resyncs"], border=True)
        st.metric("Reconnects", stats["reconnects"], border=True)
        st.metric("Parse errors", stats["parse_errors"], border=True)
        st.metric("Uptime", f"{stats['uptime_s']}s", border=True)


live_book()
