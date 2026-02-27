from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine

from ..config import Settings
from .schema import metadata


def create_engine_from_settings(settings: Settings) -> Engine:
    backend = str(settings.database.backend or "sqlite").strip().lower()
    if backend == "postgres":
        url = str(settings.database.url or "").strip()
        if not url:
            raise ValueError("database.backend=postgres requires database.url or AQF_DB_URL")
        return create_engine(url, future=True, pool_pre_ping=True)

    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def create_sqlite_engine(db_path: str | Path) -> Engine:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)
