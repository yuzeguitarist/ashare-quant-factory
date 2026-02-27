from __future__ import annotations

import io
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChartSpec:
    window: int = 140  # last N bars
    dpi: int = 140


def make_candlestick_with_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    title: str,
    spec: ChartSpec = ChartSpec(),
) -> bytes:
    """Return PNG bytes for candlestick chart with entry/exit markers."""
    if df.empty:
        return b""

    x = df.copy().sort_values("date").reset_index(drop=True)
    x = x.tail(spec.window).copy()

    # mplfinance expects datetime index
    x["date_dt"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date_dt"]).set_index("date_dt")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close"])

    # align signals
    sig = signals.copy()
    if not sig.empty:
        sig["date_dt"] = pd.to_datetime(sig["date"], errors="coerce")
        sig = sig.dropna(subset=["date_dt"]).set_index("date_dt")
        sig = sig.reindex(x.index).fillna(False)

    apds = []
    if not sig.empty:
        # Entry markers at low, exit markers at high
        entry_y = np.where(sig["entry"].to_numpy(bool), x["low"].to_numpy(float) * 0.995, np.nan)
        exit_y = np.where(sig["exit"].to_numpy(bool), x["high"].to_numpy(float) * 1.005, np.nan)
        apds.append(mpf.make_addplot(entry_y, type="scatter", markersize=50, marker="^"))
        apds.append(mpf.make_addplot(exit_y, type="scatter", markersize=50, marker="v"))

    buf = io.BytesIO()
    mpf.plot(
        x,
        type="candle",
        volume=False,
        addplot=apds if apds else None,
        title=title,
        style="yahoo",
        tight_layout=True,
        savefig=dict(fname=buf, dpi=spec.dpi, bbox_inches="tight"),
    )
    return buf.getvalue()


def make_risk_gauge(score: int, title: str = "Risk Score") -> bytes:
    """Semicircle gauge. score: 0-100."""
    score = int(max(0, min(100, score)))
    fig = plt.figure(figsize=(5.2, 3.0), dpi=160)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    # Arc background
    theta = np.linspace(math.pi, 2 * math.pi, 400)
    r_outer = 1.0
    r_inner = 0.72

    x_outer = r_outer * np.cos(theta)
    y_outer = r_outer * np.sin(theta)
    x_inner = r_inner * np.cos(theta[::-1])
    y_inner = r_inner * np.sin(theta[::-1])

    ax.fill(
        np.concatenate([x_outer, x_inner]),
        np.concatenate([y_outer, y_inner]),
        alpha=0.12,
    )

    # Ticks
    for t in range(0, 101, 10):
        ang = math.pi + (t / 100.0) * math.pi
        ax.plot([0.85 * math.cos(ang), 1.0 * math.cos(ang)], [0.85 * math.sin(ang), 1.0 * math.sin(ang)], lw=1)
        ax.text(0.70 * math.cos(ang), 0.70 * math.sin(ang), str(t), ha="center", va="center", fontsize=8, alpha=0.9)

    # Needle
    ang = math.pi + (score / 100.0) * math.pi
    ax.plot([0, 0.82 * math.cos(ang)], [0, 0.82 * math.sin(ang)], lw=3)
    ax.scatter([0], [0], s=60)

    ax.text(0, -0.10, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0, -0.28, f"{score}/100", ha="center", va="center", fontsize=22, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()
