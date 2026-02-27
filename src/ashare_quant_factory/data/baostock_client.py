from __future__ import annotations

from dataclasses import dataclass

import baostock as bs
import pandas as pd


@dataclass(frozen=True)
class BaostockSession:
    """Context-managed baostock login session."""

    def __enter__(self) -> "BaostockSession":
        lg = bs.login()
        if getattr(lg, "error_code", "1") != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_code} {lg.error_msg}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        bs.logout()

    @staticmethod
    def _to_df(rs) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        if getattr(rs, "error_code", "1") != "0":
            raise RuntimeError(f"baostock query failed: {rs.error_code} {rs.error_msg}")

        rows: list[list[str]] = []
        while rs.next():
            rows.append(rs.get_row_data())
        return pd.DataFrame(rows, columns=list(rs.fields))

    @staticmethod
    def query_trade_dates(start_date: str, end_date: str) -> pd.DataFrame:
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        return BaostockSession._to_df(rs)

    @staticmethod
    def query_history_daily(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = (
            "date,code,open,high,low,close,preclose,volume,amount,"
            "adjustflag,turn,tradestatus,pctChg,isST"
        )
        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        df = BaostockSession._to_df(rs)
        if df.empty:
            return df
        return df.drop(columns=["adjustflag"], errors="ignore")
