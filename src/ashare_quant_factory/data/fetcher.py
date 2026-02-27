from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from ..config import Settings
from ..db.repository import DB, latest_bar_date, upsert_bars
from .baostock_client import BaostockSession


@dataclass(frozen=True)
class FetchStats:
    fetched_rows: int
    inserted_rows: int


class DataFetcher:
    def __init__(self, settings: Settings, db: DB):
        self.settings = settings
        self.db = db

    def probe_daily_updated(self, trade_date: str) -> bool:
        """Return True when probe symbol has daily bar for trade_date."""
        symbol = self.settings.data.probe_symbol
        with BaostockSession() as _:
            df = BaostockSession.query_history_daily(symbol, trade_date, trade_date)
        if df.empty:
            return False

        row = df.iloc[-1]
        if str(row.get("date", "")) != trade_date:
            return False

        # Sometimes stubs may exist with empty values before full refresh.
        close_v = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
        vol_v = pd.to_numeric(pd.Series([row.get("volume")]), errors="coerce").iloc[0]
        return bool(pd.notna(close_v) and pd.notna(vol_v) and vol_v > 0)

    def sync_watchlist_incremental(self, trade_date: str) -> FetchStats:
        fetched_rows = 0
        inserted_rows = 0

        for code in self.settings.watchlist:
            start_date = self._next_date(latest_bar_date(self.db, code))
            if start_date > trade_date:
                continue

            with BaostockSession() as _:
                df = BaostockSession.query_history_daily(code, start_date=start_date, end_date=trade_date)

            if df.empty:
                continue

            fetched_rows += len(df)
            inserted_rows += upsert_bars(self.db, self._normalize_bars(df))

        return FetchStats(fetched_rows=fetched_rows, inserted_rows=inserted_rows)

    def _next_date(self, v: str | None) -> str:
        if not v:
            return self.settings.data.history_start
        return (dt.date.fromisoformat(v) + dt.timedelta(days=1)).isoformat()

    @staticmethod
    def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        keep = [
            "code",
            "date",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "tradestatus",
            "pctChg",
            "isST",
        ]
        return out[[c for c in keep if c in out.columns]].dropna(subset=["code", "date"])
