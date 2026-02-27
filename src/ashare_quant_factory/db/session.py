from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine

from .schema import metadata


def create_sqlite_engine(db_path: str | Path) -> Engine:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
