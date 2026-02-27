from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import Engine, Select, and_, func, insert, select, text
from sqlalchemy.engine import Connection

from .schema import daily_bars, ga_best, kv_store, recommendations, report_runs


@dataclass(frozen=True)
class DB:
    engine: Engine

    def connect(self) -> Connection:
        return self.engine.connect()


def get_kv(db: DB, key: str) -> str | None:
    with db.connect() as conn:
        row = conn.execute(select(kv_store.c.value).where(kv_store.c.key == key)).fetchone()
        return row[0] if row else None


def set_kv(db: DB, key: str, value: str) -> None:
    with db.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO kv_store(key, value, updated_at)
                VALUES(:key, :value, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                  value=excluded.value,
                  updated_at=CURRENT_TIMESTAMP
                """
            ),
            {"key": key, "value": value},
        )
        conn.commit()


def latest_bar_date(db: DB, code: str) -> str | None:
    with db.connect() as conn:
        row = conn.execute(select(func.max(daily_bars.c.date)).where(daily_bars.c.code == code)).fetchone()
        return row[0] if row and row[0] else None


def upsert_bars(db: DB, df: pd.DataFrame) -> int:
    """Insert bars (ignore duplicates by primary key). Return inserted count."""
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    with db.connect() as conn:
        # SQLite: INSERT OR IGNORE keeps existing rows untouched.
        result = conn.execute(insert(daily_bars).prefix_with("OR IGNORE"), records)
        conn.commit()
        return int(result.rowcount or 0)


def load_bars(db: DB, codes: Iterable[str], start_date: str | None = None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with db.connect() as conn:
        for code in codes:
            stmt: Select = select(daily_bars).where(daily_bars.c.code == code)
            if start_date:
                stmt = stmt.where(daily_bars.c.date >= start_date)
            stmt = stmt.order_by(daily_bars.c.date.asc())
            rows = conn.execute(stmt).mappings().all()
            if not rows:
                out[code] = pd.DataFrame()
                continue
            df = pd.DataFrame(rows)
            # keep columns consistent
            for c in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            out[code] = df
    return out


def save_best_strategy(db: DB, trade_date: str, genome: dict[str, Any], fitness: float, metrics: dict[str, Any]) -> None:
    with db.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO ga_best(trade_date, genome_json, fitness, metrics_json, created_at)
                VALUES(:trade_date, :genome_json, :fitness, :metrics_json, CURRENT_TIMESTAMP)
                ON CONFLICT(trade_date) DO UPDATE SET
                    genome_json=excluded.genome_json,
                    fitness=excluded.fitness,
                    metrics_json=excluded.metrics_json,
                    created_at=CURRENT_TIMESTAMP
                """
            ),
            {
                "trade_date": trade_date,
                "genome_json": json.dumps(genome, ensure_ascii=False),
                "fitness": float(fitness),
                "metrics_json": json.dumps(metrics, ensure_ascii=False),
            },
        )
        conn.commit()


def load_best_strategy(db: DB, trade_date: str) -> dict[str, Any] | None:
    with db.connect() as conn:
        row = conn.execute(select(ga_best).where(ga_best.c.trade_date == trade_date)).mappings().fetchone()
        if not row:
            return None
        return {
            "trade_date": row["trade_date"],
            "genome": json.loads(row["genome_json"]),
            "fitness": row["fitness"],
            "metrics": json.loads(row["metrics_json"]),
        }


def save_recommendations(db: DB, trade_date: str, recos: pd.DataFrame) -> None:
    if recos.empty:
        return
    required = {"code", "action", "weight", "stop_loss", "take_profit", "expected_return", "risk_score", "note"}
    missing = required - set(recos.columns)
    if missing:
        raise ValueError(f"recommendations df missing columns: {sorted(missing)}")

    records = [{"trade_date": trade_date, **r} for r in recos.to_dict(orient="records")]
    with db.connect() as conn:
        # Upsert by (trade_date, code)
        conn.execute(text("DELETE FROM recommendations WHERE trade_date=:d"), {"d": trade_date})
        conn.execute(insert(recommendations), records)
        conn.commit()


def load_recommendations(db: DB, trade_date: str) -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute(select(recommendations).where(recommendations.c.trade_date == trade_date)).mappings().all()
        return pd.DataFrame(rows) if rows else pd.DataFrame()


def save_report_run(db: DB, trade_date: str, html_path: str, sent_to: str) -> None:
    with db.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO report_runs(trade_date, html_path, sent_to, sent_at)
                VALUES(:d, :p, :to, CURRENT_TIMESTAMP)
                ON CONFLICT(trade_date) DO UPDATE SET
                    html_path=excluded.html_path,
                    sent_to=excluded.sent_to,
                    sent_at=CURRENT_TIMESTAMP
                """
            ),
            {"d": trade_date, "p": html_path, "to": sent_to},
        )
        conn.commit()
