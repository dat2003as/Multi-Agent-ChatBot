"""
app/vanna/executor.py
=====================
SQL Executor — A2 Inventory WH

Theo diagram: "Thực thi SQL → PostgreSQL (Execute validated query)"
  - Nhận SQL đã qua guardrail
  - Kết nối Backend DB (PostgreSQL + RLS)
  - Execute và trả về data dạng list[dict]
  - Bắt exception → trả lên Error Capture layer

Pipeline position: sau guardrail, trước trả kết quả về Orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any
from decimal import Decimal
from datetime import datetime, date
import re

import psycopg2
import psycopg2.extras

from app.core.db_pool import get_be_connection, return_be_connection

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MAX_ROWS = 500   # Giới hạn số dòng trả về — tránh query quá nặng


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────
class ExecutorError(Exception):
    """Raised khi execute SQL thất bại."""


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZATION HELPERS (JSON-safe conversion)
# ─────────────────────────────────────────────────────────────────────────────
def _convert_decimals(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable types to JSON-safe types.
    
    - Decimal → float
    - date/datetime → ISO format string
    - dict/list → recurse
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_decimals(item) for item in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def execute_sql(
    sql: str,
    farm_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Thực thi SQL trên Backend DB (PostgreSQL + RLS).

    Args:
        sql:     SQL đã được validate bởi guardrail.
        farm_id: UUID trang trại — dùng để set RLS context nếu cần.

    Returns:
        Tuple (data, columns):
          - data:    list[dict] — mỗi row là một dict {column: value}
          - columns: list[str] — danh sách tên cột theo thứ tự

    Raises:
        ExecutorError: Khi query thất bại.
    """
    conn = None
    try:
        conn = get_be_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Set RLS context (farm isolation tại DB layer)
        _set_rls_context(cur, farm_id)

        # Execute query với LIMIT bảo vệ
        safe_sql = _apply_row_limit(sql)
        logger.debug("[Executor] running SQL:\n%s", safe_sql)

        cur.execute(safe_sql)
        rows = cur.fetchall()

        # Lấy tên cột
        columns = [desc[0] for desc in cur.description] if cur.description else []

        # Convert RealDictRow → plain dict & sanitize for JSON (Decimal → float, datetime → ISO)
        data = [_convert_decimals(dict(row)) for row in rows]

        logger.info("[Executor] query OK rows=%d cols=%d", len(data), len(columns))
        cur.close()
        return data, columns

    except psycopg2.Error as exc:
        logger.error("[Executor] PostgreSQL error: %s", exc)
        raise ExecutorError(f"Database error: {exc.pgerror or str(exc)}") from exc

    except Exception as exc:  # noqa: BLE001
        logger.error("[Executor] unexpected error: %s", exc)
        raise ExecutorError(f"Executor unexpected error: {exc}") from exc

    finally:
        return_be_connection(conn)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _set_rls_context(cur: psycopg2.extensions.cursor, farm_id: str) -> None:
    """
    Set Row Level Security context cho session.

    Nếu BE DB dùng RLS (theo diagram: PostgreSQL + RLS),
    cần set app.current_farm_id để PostgreSQL policies hoạt động.
    """
    try:
        cur.execute(
            "SELECT set_config('app.current_farm_id', %s, true)",
            (farm_id,),
        )
        logger.debug("[Executor] RLS context set farm_id=%s", farm_id)
    except psycopg2.Error:
        # Nếu config key chưa được khai báo trong DB → bỏ qua
        # (farm isolation đã được guardrail + WHERE ZoneId xử lý)
        logger.debug("[Executor] RLS set_config skipped (not configured in DB)")


def _apply_row_limit(sql: str) -> str:
    """
    Thêm LIMIT nếu SQL chưa có, để tránh query trả về quá nhiều dòng.

    Chỉ áp dụng cho top-level SELECT, không wrap vào subquery.
    """
    sql_upper = sql.upper().strip()

    # Nếu đã có LIMIT → giữ nguyên
    if re.search(r"\bLIMIT\s+\d+", sql_upper):
        return sql

    # Nếu có ORDER BY → thêm LIMIT trước dấu ; hoặc cuối chuỗi
    sql = sql.rstrip().rstrip(";")
    sql = f"{sql}\nLIMIT {MAX_ROWS}"
    logger.debug("[Executor] auto-applied LIMIT %d", MAX_ROWS)
    return sql