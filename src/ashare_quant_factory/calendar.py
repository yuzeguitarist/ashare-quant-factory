from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .data.baostock_client import BaostockSession
from .timeutils import parse_yyyy_mm_dd, to_date_str, tz_now


@dataclass(frozen=True)
class TradeDay:
    last: str  # last trading date <= today (YYYY-MM-DD)
    next: str  # next trading date > last (YYYY-MM-DD)


def get_trade_day(today: dt.date) -> TradeDay:
    """Get last & next trade day around `today` by querying Baostock trade calendar."""
    start = to_date_str(today - dt.timedelta(days=40))
    end = to_date_str(today + dt.timedelta(days=40))
    with BaostockSession() as _:
        cal = BaostockSession.query_trade_dates(start_date=start, end_date=end)

    if cal.empty:
        raise RuntimeError("Empty trade calendar from baostock")

    cal = cal.rename(columns={"calendar_date": "date"}).copy()
    cal["is_trading_day"] = cal["is_trading_day"].astype(str)

    # last trade day <= today
    cal["date_dt"] = pd.to_datetime(cal["date"], errors="coerce")
    cal = cal.dropna(subset=["date_dt"]).sort_values("date_dt")

    today_ts = pd.Timestamp(today)

    past = cal[(cal["date_dt"] <= today_ts) & (cal["is_trading_day"] == "1")]
    if past.empty:
        raise RuntimeError(f"No trading day found before {today}")

    last = str(past.iloc[-1]["date"])  # YYYY-MM-DD

    future = cal[(cal["date_dt"] > pd.Timestamp(parse_yyyy_mm_dd(last))) & (cal["is_trading_day"] == "1")]
    if future.empty:
        # extremely rare (end range too short) -> extend
        raise RuntimeError("No next trading day found; please extend calendar range")
    nxt = str(future.iloc[0]["date"])
    return TradeDay(last=last, next=nxt)


def get_trade_day_now(tz_name: str = "Asia/Shanghai") -> TradeDay:
    return get_trade_day(tz_now(tz_name).date())
