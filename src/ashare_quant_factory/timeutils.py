from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketTimes:
    tz: ZoneInfo
    poll_start: dt.time
    open_time: dt.time
    stop_before_open: dt.timedelta


def parse_hhmm(value: str) -> dt.time:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {value}")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid HH:MM time: {value}")
    return dt.time(hour=h, minute=m)


def tz_now(tz_name: str) -> dt.datetime:
    return dt.datetime.now(tz=ZoneInfo(tz_name))


def combine_local(date_: dt.date, t: dt.time, tz_name: str) -> dt.datetime:
    tz = ZoneInfo(tz_name)
    return dt.datetime.combine(date_, t).replace(tzinfo=tz)


def to_date_str(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def to_yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def parse_yyyy_mm_dd(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()
