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

    x["date_dt"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date_dt"]).set_index("date_dt")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close"])

    sig = signals.copy()
    if not sig.empty:
        sig["date_dt"] = pd.to_datetime(sig["date"], errors="coerce")
        sig = sig.dropna(subset=["date_dt"]).set_index("date_dt")
        sig = sig.reindex(x.index).fillna(False)

    apds = []
    if not sig.empty:
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
    score = int(max(0, min(100, score)))
    fig = plt.figure(figsize=(5.2, 3.0), dpi=160)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    theta = np.linspace(math.pi, 2 * math.pi, 400)
    r_outer = 1.0
    r_inner = 0.72

    x_outer = r_outer * np.cos(theta)
    y_outer = r_outer * np.sin(theta)
    x_inner = r_inner * np.cos(theta[::-1])
    y_inner = r_inner * np.sin(theta[::-1])

    ax.fill(np.concatenate([x_outer, x_inner]), np.concatenate([y_outer, y_inner]), alpha=0.12)

    for t in range(0, 101, 10):
        ang = math.pi + (t / 100.0) * math.pi
        ax.plot([0.85 * math.cos(ang), 1.0 * math.cos(ang)], [0.85 * math.sin(ang), 1.0 * math.sin(ang)], lw=1)
        ax.text(0.70 * math.cos(ang), 0.70 * math.sin(ang), str(t), ha="center", va="center", fontsize=8, alpha=0.9)

    ang = math.pi + (score / 100.0) * math.pi
    ax.plot([0, 0.82 * math.cos(ang)], [0, 0.82 * math.sin(ang)], lw=3)
    ax.scatter([0], [0], s=60)

    ax.text(0, -0.10, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0, -0.28, f"{score}/100", ha="center", va="center", fontsize=22, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def make_stability_card(stability_score: int, cv_sharpe_std: float) -> bytes:
    fig, ax = plt.subplots(figsize=(4.8, 2.4), dpi=160)
    ax.axis("off")
    score = max(0, min(100, int(stability_score)))
    ax.barh([0], [100], color="#1f2937", alpha=0.35, height=0.45)
    ax.barh([0], [score], color="#22c55e" if score >= 70 else "#f59e0b", height=0.45)
    ax.text(1, 0.35, f"稳定性评分: {score}/100", fontsize=12, fontweight="bold", color="#e5e7eb")
    ax.text(1, -0.35, f"CV Sharpe 标准差: {cv_sharpe_std:.3f}", fontsize=10, color="#cbd5e1")
    ax.set_xlim(0, 100)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def make_param_sensitivity_chart(items: list[dict[str, float]]) -> bytes:
    if not items:
        return b""
    top = items[:6]
    names = [str(x.get("param", "")) for x in top]
    vals = [float(x.get("impact", 0.0)) for x in top]

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=160)
    y = np.arange(len(names))
    ax.barh(y, vals, color="#60a5fa", alpha=0.9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("fitness impact")
    ax.set_title("参数敏感性（扰动后 fitness 变化）", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()
