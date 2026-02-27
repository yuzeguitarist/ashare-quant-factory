from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()

daily_bars = Table(
    "daily_bars",
    metadata,
    Column("code", String(16), primary_key=True),
    Column("date", String(16), primary_key=True),  # YYYY-MM-DD
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("preclose", Float),
    Column("volume", Float),
    Column("amount", Float),
    Column("turn", Float),
    Column("tradestatus", String(8)),
    Column("pctChg", Float),
    Column("isST", String(8)),
    Column("created_at", DateTime, default=dt.datetime.utcnow, nullable=False),
)

kv_store = Table(
    "kv_store",
    metadata,
    Column("key", String(64), primary_key=True),
    Column("value", Text, nullable=False),
    Column("updated_at", DateTime, default=dt.datetime.utcnow, nullable=False),
)

ga_best = Table(
    "ga_best",
    metadata,
    Column("trade_date", String(16), primary_key=True),
    Column("genome_json", Text, nullable=False),
    Column("fitness", Float, nullable=False),
    Column("metrics_json", Text, nullable=False),
    Column("created_at", DateTime, default=dt.datetime.utcnow, nullable=False),
)

recommendations = Table(
    "recommendations",
    metadata,
    Column("trade_date", String(16), primary_key=True),
    Column("code", String(16), primary_key=True),
    Column("action", String(8), nullable=False),  # BUY/SELL/HOLD
    Column("weight", Float, nullable=False),
    Column("stop_loss", Float),
    Column("take_profit", Float),
    Column("expected_return", Float),
    Column("risk_score", Integer),
    Column("note", Text),
    Column("created_at", DateTime, default=dt.datetime.utcnow, nullable=False),
    UniqueConstraint("trade_date", "code", name="uq_reco_trade_date_code"),
)

report_runs = Table(
    "report_runs",
    metadata,
    Column("trade_date", String(16), primary_key=True),
    Column("html_path", Text, nullable=False),
    Column("sent_to", Text, nullable=False),
    Column("sent_at", DateTime, default=dt.datetime.utcnow, nullable=False),
)
