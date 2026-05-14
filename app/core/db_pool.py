"""
Shared psycopg2 connection pool for Backend DB (BE_DB_URL).

Used by:
  - app/vanna/executor.py (SQL execution)
  - app/api/farm_context.py (farm name resolution)
"""
from __future__ import annotations

import logging
import atexit

import psycopg2.pool

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_be_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=settings.BE_DB_URL,
            connect_timeout=10,
        )
        atexit.register(_close_pool)
        logger.info("[DBPool] ThreadedConnectionPool created (min=2, max=10)")
    return _pool


def get_be_connection() -> psycopg2.extensions.connection:
    """Get a connection from the pool. Caller MUST return it via return_be_connection()."""
    pool = get_be_pool()
    conn = pool.getconn()
    conn.set_session(readonly=True, autocommit=False)
    return conn


def return_be_connection(conn: psycopg2.extensions.connection) -> None:
    """Return a connection back to the pool."""
    if conn is None:
        return
    try:
        conn.rollback()
    except Exception:
        pass
    try:
        pool = get_be_pool()
        pool.putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def _close_pool() -> None:
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        logger.info("[DBPool] Connection pool closed")
    _pool = None
