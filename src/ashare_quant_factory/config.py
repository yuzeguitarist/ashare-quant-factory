from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from dotenv import load_dotenv


def _as_path(v: str | Path) -> Path:
    return v if isinstance(v, Path) else Path(v)


def _split_csv(v: str) -> list[str]:
    return [s.strip() for s in v.split(",") if s.strip()]


def _deep_update(base: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base


@dataclass(frozen=True)
class Project:
    name: str = "AShare Quant Factory"
    timezone: str = "Asia/Shanghai"


@dataclass(frozen=True)
class Paths:
    db: str = "data/aqf.sqlite3"
    db_url: str = ""
    reports_dir: str = "data/reports"
    lock_file: str = "data/aqf.lock"


@dataclass(frozen=True)
class Data:
    history_start: str = "2016-01-01"
    probe_symbol: str = "sh.000001"


@dataclass(frozen=True)
class Schedule:
    poll_start_time: str = "20:00"
    poll_interval_seconds: int = 300
    market_open_time: str = "09:30"
    stop_minutes_before_open: int = 30


@dataclass(frozen=True)
class Backtest:
    commission_bps: float = 3.0
    stamp_duty_bps: float = 10.0
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class GA:
    population_size: int = 64
    elite_size: int = 8
    crossover_rate: float = 0.65
    mutation_rate: float = 0.25
    workers: int = 3
    max_eval_symbols: int = 0
    seed: int | None = 42
    cv_mode: str = "plain"  # plain|walk_forward|purged_kfold
    cv_splits: int = 4
    cv_purge_days: int = 3


@dataclass(frozen=True)
class Risk:
    max_weight_per_symbol: float = 0.20
    top_charts: int = 6


@dataclass(frozen=True)
class Email:
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_name: str = "AQF Nightly Alpha"
    to: tuple[str, ...] = ()
    subject_prefix: str = "[AQF] "


@dataclass(frozen=True)
class Settings:
    project: Project
    paths: Paths
    data: Data
    schedule: Schedule
    watchlist: tuple[str, ...]
    backtest: Backtest
    ga: GA
    risk: Risk
    email: Email

    @property
    def db_path(self) -> Path:
        return _as_path(self.paths.db)

    @property
    def db_url(self) -> str:
        return str(self.paths.db_url or "").strip()

    @property
    def reports_dir(self) -> Path:
        return _as_path(self.paths.reports_dir)

    @property
    def lock_file(self) -> Path:
        return _as_path(self.paths.lock_file)


def load_settings(config_path: str | Path, env_path: str | Path | None = None) -> Settings:
    """Load YAML settings + .env secrets + env overrides.

    Args:
        config_path: config.yaml path
        env_path: optional .env path (default: .env next to config, then ./ .env)
    """
    config_path = _as_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load .env (optional)
    candidates: list[Path] = []
    if env_path is not None:
        candidates.append(_as_path(env_path))
    candidates.append(config_path.with_suffix(".env"))  # config.env
    candidates.append(Path(".env"))
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            break

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("config.yaml must be a mapping/dict at root")

    defaults: dict[str, Any] = {
        "project": {"name": Project.name, "timezone": Project.timezone},
        "paths": {
            "db": Paths.db,
            "db_url": Paths.db_url,
            "reports_dir": Paths.reports_dir,
            "lock_file": Paths.lock_file,
        },
        "data": {"history_start": Data.history_start, "probe_symbol": Data.probe_symbol},
        "schedule": {
            "poll_start_time": Schedule.poll_start_time,
            "poll_interval_seconds": Schedule.poll_interval_seconds,
            "market_open_time": Schedule.market_open_time,
            "stop_minutes_before_open": Schedule.stop_minutes_before_open,
        },
        "watchlist": [],
        "backtest": {
            "commission_bps": Backtest.commission_bps,
            "stamp_duty_bps": Backtest.stamp_duty_bps,
            "slippage_bps": Backtest.slippage_bps,
        },
        "ga": {
            "population_size": GA.population_size,
            "elite_size": GA.elite_size,
            "crossover_rate": GA.crossover_rate,
            "mutation_rate": GA.mutation_rate,
            "workers": GA.workers,
            "max_eval_symbols": GA.max_eval_symbols,
            "seed": GA.seed,
            "cv_mode": GA.cv_mode,
            "cv_splits": GA.cv_splits,
            "cv_purge_days": GA.cv_purge_days,
        },
        "risk": {"max_weight_per_symbol": Risk.max_weight_per_symbol, "top_charts": Risk.top_charts},
        "email": {
            "smtp_host": Email.smtp_host,
            "smtp_port": Email.smtp_port,
            "sender_name": Email.sender_name,
            "to": [],
            "subject_prefix": Email.subject_prefix,
        },
    }

    merged = _deep_update(defaults, raw)

    # Env overrides for email
    smtp_host = os.getenv("AQF_SMTP_HOST")
    smtp_port = os.getenv("AQF_SMTP_PORT")
    email_to = os.getenv("AQF_EMAIL_TO")
    if smtp_host:
        merged["email"]["smtp_host"] = smtp_host
    if smtp_port:
        merged["email"]["smtp_port"] = int(smtp_port)
    if email_to:
        merged["email"]["to"] = _split_csv(email_to)

    watchlist = tuple(str(x).strip() for x in merged.get("watchlist", []) if str(x).strip())
    if not watchlist:
        raise ValueError(
            "watchlist is empty. Please set `watchlist` in config.yaml (Baostock codes like sh.600519)."
        )

    return Settings(
        project=Project(**merged["project"]),
        paths=Paths(**merged["paths"]),
        data=Data(**merged["data"]),
        schedule=Schedule(**merged["schedule"]),
        watchlist=watchlist,
        backtest=Backtest(**merged["backtest"]),
        ga=GA(**merged["ga"]),
        risk=Risk(**merged["risk"]),
        email=Email(
            smtp_host=merged["email"]["smtp_host"],
            smtp_port=int(merged["email"]["smtp_port"]),
            sender_name=merged["email"]["sender_name"],
            to=tuple(merged["email"].get("to") or ()),
            subject_prefix=merged["email"]["subject_prefix"],
        ),
    )


def iter_watchlist(settings: Settings) -> Iterable[str]:
    return settings.watchlist
