from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine

from .schema import metadata


def create_sqlite_engine(db_path: str | Path) -> Engine:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


def create_db_engine(db_path: str | Path, db_url: str = "") -> Engine:
    if db_url.strip():
        return create_engine(db_url.strip(), future=True, pool_pre_ping=True)
    return create_sqlite_engine(db_path)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
