from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .indicators import atr_wilder, rsi_wilder, sma


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 3.0
    stamp_duty_bps: float = 10.0
    slippage_bps: float = 2.0

    @property
    def buy_cost(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10000.0

    @property
    def sell_cost(self) -> float:
        return (self.commission_bps + self.slippage_bps + self.stamp_duty_bps) / 10000.0


@dataclass(frozen=True)
class Genome:
    fast_ma: int
    slow_ma: int
    rsi_period: int
    rsi_buy: float
    rsi_sell: float
    atr_period: int
    stop_loss_atr: float
    take_profit_atr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_ma": int(self.fast_ma),
            "slow_ma": int(self.slow_ma),
            "rsi_period": int(self.rsi_period),
            "rsi_buy": float(self.rsi_buy),
            "rsi_sell": float(self.rsi_sell),
            "atr_period": int(self.atr_period),
            "stop_loss_atr": float(self.stop_loss_atr),
            "take_profit_atr": float(self.take_profit_atr),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Genome":
        return Genome(
            fast_ma=int(d["fast_ma"]),
            slow_ma=int(d["slow_ma"]),
            rsi_period=int(d["rsi_period"]),
            rsi_buy=float(d["rsi_buy"]),
            rsi_sell=float(d["rsi_sell"]),
            atr_period=int(d["atr_period"]),
            stop_loss_atr=float(d["stop_loss_atr"]),
            take_profit_atr=float(d["take_profit_atr"]),
        )


@dataclass(frozen=True)
class BacktestMetrics:
    n_days: int
    n_trades: int
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float


@dataclass(frozen=True)
class BacktestResult:
    metrics: BacktestMetrics
    equity_curve: pd.DataFrame  # date, equity
    signals: pd.DataFrame  # date, entry, exit
    trades: pd.DataFrame  # entry_date, exit_date, entry_price, exit_price, ret


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(dd.min())  # negative


def _sharpe(daily_returns: np.ndarray) -> float:
    if daily_returns.size < 10:
        return 0.0
    mu = float(np.mean(daily_returns))
    sigma = float(np.std(daily_returns, ddof=1))
    if sigma <= 1e-12:
        return 0.0
    return math.sqrt(252.0) * mu / sigma


def generate_signals(df: pd.DataFrame, g: Genome) -> pd.DataFrame:
    """Generate entry/exit signals on close-of-day basis."""
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    x = x.sort_values("date").reset_index(drop=True)
    if "tradestatus" in x.columns:
        x = x[x["tradestatus"].astype(str) == "1"].reset_index(drop=True)

    close = x["close"].astype(float)
    fast = sma(close, g.fast_ma)
    slow = sma(close, g.slow_ma)
    rsi = rsi_wilder(close, g.rsi_period)

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    entry = cross_up & (rsi < g.rsi_buy)
    exit_ = cross_dn | (rsi > g.rsi_sell)

    sig = pd.DataFrame({"date": x["date"], "entry": entry.fillna(False), "exit": exit_.fillna(False)})
    return sig


def backtest(df: pd.DataFrame, g: Genome, costs: CostModel) -> BacktestResult:
    """Simplified A-share daily backtest.

    Assumptions:
    - Signals are computed on day t close, executed on day t+1 open (T+1 compatible).
    - Stop-loss / take-profit are checked on day t close (not intraday) and executed next day open.
    - 100% capital per symbol (single-asset backtest).
    """
    if df.empty or len(df) < max(g.slow_ma, g.atr_period, g.rsi_period) + 5:
        empty_metrics = BacktestMetrics(
            n_days=0,
            n_trades=0,
            total_return=0.0,
            annual_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
        )
        return BacktestResult(empty_metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    x = df.copy().sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close"])
    x = x.reset_index(drop=True)
    if "tradestatus" in x.columns:
        x = x[x["tradestatus"].astype(str) == "1"].reset_index(drop=True)

    sig = generate_signals(x, g)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    close = x["close"].astype(float)
    open_ = x["open"].astype(float)

    atr = atr_wilder(high, low, close, g.atr_period)

    n = len(x)
    equity = 1.0
    equity_series: list[float] = [equity]
    eq_dates: list[str] = [str(x.loc[0, "date"])]

    in_pos = False
    entry_price = 0.0
    entry_atr = 0.0
    entry_date = ""
    trades: list[dict[str, Any]] = []
    daily_rets: list[float] = []

    # loop from day 1 since we execute on open[t] based on signal[t-1]
    for t in range(1, n):
        prev = t - 1
        prev_close = float(close.iloc[prev])
        today_open = float(open_.iloc[t])
        today_close = float(close.iloc[t])

        # By default, cash: no exposure
        day_ret = 0.0

        if in_pos:
            stop_px = entry_price - entry_atr * g.stop_loss_atr
            take_px = entry_price + entry_atr * g.take_profit_atr

            should_exit = bool(sig.loc[prev, "exit"]) or (prev_close <= stop_px) or (prev_close >= take_px)
            if should_exit:
                exit_price = today_open * (1.0 - costs.sell_cost)
                day_ret = (exit_price / prev_close) - 1.0
                equity *= 1.0 + day_ret

                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": str(x.loc[t, "date"]),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "ret": (exit_price / entry_price) - 1.0,
                        "reason": "signal/stop/take",
                    }
                )
                in_pos = False
                entry_price = 0.0
                entry_atr = 0.0
                entry_date = ""
            else:
                # hold through close
                day_ret = (today_close / prev_close) - 1.0
                equity *= 1.0 + day_ret

        # only consider entry if we are flat
        if not in_pos:
            if bool(sig.loc[prev, "entry"]):
                entry_price = today_open * (1.0 + costs.buy_cost)
                entry_atr = float(atr.iloc[prev]) if pd.notna(atr.iloc[prev]) else 0.0
                entry_date = str(x.loc[t, "date"])

                # exposure from open->close on entry day
                day_ret = (today_close / entry_price) - 1.0
                equity *= 1.0 + day_ret
                in_pos = True

        equity_series.append(equity)
        eq_dates.append(str(x.loc[t, "date"]))
        daily_rets.append(day_ret)

    equity_arr = np.asarray(equity_series, dtype=float)
    dd = _max_drawdown(equity_arr)
    sharpe = _sharpe(np.asarray(daily_rets, dtype=float))

    n_days = max(0, len(daily_rets))
    if n_days > 0:
        annual_ret = equity ** (252.0 / n_days) - 1.0
    else:
        annual_ret = 0.0

    trade_rets = np.asarray([t["ret"] for t in trades], dtype=float)
    n_trades = int(trade_rets.size)
    win_rate = float((trade_rets > 0).mean()) if n_trades > 0 else 0.0
    pos = trade_rets[trade_rets > 0].sum() if n_trades > 0 else 0.0
    neg = trade_rets[trade_rets < 0].sum() if n_trades > 0 else 0.0
    profit_factor = float(pos / abs(neg)) if neg < 0 else float("inf") if pos > 0 else 0.0

    metrics = BacktestMetrics(
        n_days=n_days,
        n_trades=n_trades,
        total_return=float(equity - 1.0),
        annual_return=float(annual_ret),
        sharpe=float(sharpe),
        max_drawdown=float(dd),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor if math.isfinite(profit_factor) else 0.0),
    )

    equity_curve = pd.DataFrame({"date": eq_dates, "equity": equity_series})
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "entry_price", "exit_price", "ret", "reason"]
    )
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, signals=sig, trades=trades_df)