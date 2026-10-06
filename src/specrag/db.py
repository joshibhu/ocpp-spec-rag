"""Database access. One pool, parameterised queries."""

from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .settings import settings

pool = ConnectionPool(settings.database_url, min_size=1, max_size=5, open=False)


def _ensure_open() -> None:
    if pool.closed:
        pool.open(wait=True, timeout=10.0)


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    _ensure_open()
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall() if cur.description else []


def execute(sql: str, params: tuple = ()) -> None:
    _ensure_open()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)


def executemany(sql: str, rows: list[tuple]) -> None:
    """One statement, many parameter sets, one transaction."""
    _ensure_open()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)


def healthy() -> bool:
    try:
        return query("SELECT 1 AS ok")[0]["ok"] == 1
    except Exception:
        return False
