from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .backtester import CostModel, Genome, backtest, generate_signals
from .indicators import atr_wilder


@dataclass(frozen=True)
class Recommendation:
    code: str
    action: str  # BUY/SELL/HOLD
    weight: float
    stop_loss: float | None
    take_profit: float | None
    expected_return: float | None
    risk_score: int | None
    note: str


def _annualized_vol(close: pd.Series, window: int = 120) -> float:
    x = close.astype(float).pct_change().dropna()
    if len(x) < 20:
        return 0.0
    x = x.tail(window)
    return float(x.std(ddof=1) * math.sqrt(252.0))


def _cap_normalize(weights: np.ndarray, cap: float) -> np.ndarray:
    if weights.size == 0:
        return weights
    w = weights.copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum()

    # iterative cap
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.any():
            w[under] += excess * (w[under] / w[under].sum())
        else:
            break
    # final normalize
    if w.sum() > 0:
        w = w / w.sum()
    return w


def _risk_score(vol: float, dd: float) -> int:
    """Risk score 0-100. Higher means riskier."""
    vol_ref = 0.40  # 40% annualized vol ~ high risk
    dd_ref = 0.30   # 30% drawdown ~ high risk
    v = min(vol / vol_ref, 1.0) if vol_ref > 0 else 0.0
    d = min(abs(dd) / dd_ref, 1.0) if dd_ref > 0 else 0.0
    score = 100.0 * (0.6 * v + 0.4 * d)
    return int(round(score))


def _infer_position_state(df: pd.DataFrame, g: Genome, costs: CostModel) -> dict[str, Any]:
    """Infer whether strategy is in position at the end of the series."""
    if df.empty:
        return {"in_pos": False}

    x = df.copy().sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if "tradestatus" in x.columns:
        x = x[x["tradestatus"].astype(str) == "1"].reset_index(drop=True)
    if len(x) < 5:
        return {"in_pos": False}

    sig = generate_signals(x, g)
    atr = atr_wilder(x["high"].astype(float), x["low"].astype(float), x["close"].astype(float), g.atr_period)

    in_pos = False
    entry_price = 0.0
    entry_atr = 0.0
    entry_date = ""

    # simulate same as backtest, but only state
    for t in range(1, len(x)):
        prev = t - 1
        prev_close = float(x.loc[prev, "close"])
        today_open = float(x.loc[t, "open"])

        if in_pos:
            stop_px = entry_price - entry_atr * g.stop_loss_atr
            take_px = entry_price + entry_atr * g.take_profit_atr
            should_exit = bool(sig.loc[prev, "exit"]) or (prev_close <= stop_px) or (prev_close >= take_px)
            if should_exit:
                # exit at open[t]
                in_pos = False
                entry_price = 0.0
                entry_atr = 0.0
                entry_date = ""

        if not in_pos and bool(sig.loc[prev, "entry"]):
            entry_price = today_open * (1.0 + costs.buy_cost)
            entry_atr = float(atr.iloc[prev]) if pd.notna(atr.iloc[prev]) else 0.0
            entry_date = str(x.loc[t, "date"])
            in_pos = True

    return {"in_pos": in_pos, "entry_price": entry_price, "entry_atr": entry_atr, "entry_date": entry_date}


def make_recommendations(
    dataset: dict[str, pd.DataFrame],
    g: Genome,
    costs: CostModel,
    max_weight_per_symbol: float = 0.20,
) -> pd.DataFrame:
    recs: list[Recommendation] = []

    scores: list[float] = []
    tmp: list[dict[str, Any]] = []

    for code, df in dataset.items():
        if df.empty:
            recs.append(
                Recommendation(
                    code=code,
                    action="HOLD",
                    weight=0.0,
                    stop_loss=None,
                    take_profit=None,
                    expected_return=None,
                    risk_score=None,
                    note="No data in DB",
                )
            )
            scores.append(0.0)
            continue

        r = backtest(df, g, costs)
        state = _infer_position_state(df, g, costs)

        # latest signal is based on last bar close -> action for next open
        sig = r.signals
        if sig.empty:
            action = "HOLD"
            note = "Insufficient data for indicators"
        else:
            last_sig = sig.iloc[-1]
            if bool(last_sig["entry"]) and not state.get("in_pos", False):
                action = "BUY"
                note = "Entry signal (MA cross + RSI)"
            elif state.get("in_pos", False) and bool(last_sig["exit"]):
                action = "SELL"
                note = "Exit signal (MA cross/RSI)"
            elif state.get("in_pos", False):
                action = "HOLD"
                note = "In position"
            else:
                action = "HOLD"
                note = "No action"

        close = df.sort_values("date")["close"].astype(float)
        vol = _annualized_vol(close)
        dd = r.metrics.max_drawdown
        risk = _risk_score(vol, dd)

        exp = None
        if r.trades is not None and not r.trades.empty:
            exp = float(pd.to_numeric(r.trades["ret"], errors="coerce").dropna().mean())

        # stop/take based on last ATR and last close (or entry price if holding)
        x = df.sort_values("date").reset_index(drop=True)
        for col in ["high", "low", "close"]:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        atr = atr_wilder(x["high"], x["low"], x["close"], g.atr_period)
        last_atr = float(atr.iloc[-1]) if len(atr) else 0.0
        last_close = float(x.loc[len(x) - 1, "close"])
        anchor = float(state.get("entry_price") or last_close)

        stop = anchor - last_atr * g.stop_loss_atr if last_atr > 0 else None
        take = anchor + last_atr * g.take_profit_atr if last_atr > 0 else None

        # score for weight allocation (only BUY/HOLD-in-pos candidates get capital)
        score = 0.0
        if action in {"BUY", "HOLD"} and state.get("in_pos", False):
            # holding positions also compete for capital
            score = max(0.0, (exp or 0.0)) / max(1e-6, vol)
        elif action == "BUY":
            score = max(0.0, (exp or 0.0)) / max(1e-6, vol)
        else:
            score = 0.0

        tmp.append(
            {
                "code": code,
                "action": action,
                "stop_loss": stop,
                "take_profit": take,
                "expected_return": exp,
                "risk_score": risk,
                "note": note,
                "_score": score,
            }
        )
        scores.append(score)

    # allocate weights
    w_raw = np.asarray(scores, dtype=float)
    if w_raw.sum() > 0:
        w = _cap_normalize(w_raw, max_weight_per_symbol)
    else:
        # fallback: equal weight across BUY candidates (avoid all-zero portfolio)
        w = np.zeros_like(w_raw)
        buy_idx = [i for i, r in enumerate(tmp) if r.get("action") == "BUY"]
        if buy_idx:
            w[buy_idx] = 1.0 / len(buy_idx)
            w = _cap_normalize(w, max_weight_per_symbol)

    out_rows: list[dict[str, Any]] = []
    for row, weight in zip(tmp, w, strict=False):
        if row["action"] == "SELL":
            weight = 0.0
        out_rows.append(
            {
                "code": row["code"],
                "action": row["action"],
                "weight": float(weight),
                "stop_loss": float(row["stop_loss"]) if row["stop_loss"] is not None else None,
                "take_profit": float(row["take_profit"]) if row["take_profit"] is not None else None,
                "expected_return": float(row["expected_return"]) if row["expected_return"] is not None else None,
                "risk_score": int(row["risk_score"]) if row["risk_score"] is not None else None,
                "note": row["note"],
            }
        )

    recos = pd.DataFrame(out_rows)
    # sort by action then weight
    action_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    recos["_ao"] = recos["action"].map(action_order).fillna(9)
    recos = recos.sort_values(["_ao", "weight"], ascending=[True, False]).drop(columns=["_ao"])
    return recos