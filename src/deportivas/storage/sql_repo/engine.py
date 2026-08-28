"""SQLAlchemy engine, cached per process."""

from __future__ import annotations

import functools

from sqlalchemy import Engine, create_engine

from deportivas.config.settings import get_settings


@functools.lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, future=True)
