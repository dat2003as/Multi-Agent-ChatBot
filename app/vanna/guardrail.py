"""
app/vanna/guardrail.py
======================
SQL Guardrail — A2 Inventory WH

Theo diagram: "SQL Guardrail + farm_id Validator"
  - Chỉ SELECT
  - Inject WHERE farm_id (ZoneId)
  - Block cross-farm
  - IsDeleted = false check

Pipeline position: sau generate SQL, trước execute SQL.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────
class GuardrailError(Exception):
    """Raised khi SQL vi phạm guardrail rules."""


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Các từ khóa DML/DDL bị cấm tuyệt đối
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE,
)

# SQL injection patterns phổ biến
_INJECTION_PATTERNS = re.compile(
    r"(--|;.*;|/\*|\*/|xp_|UNION\s+ALL\s+SELECT|INTO\s+OUTFILE|LOAD_FILE)",
    re.IGNORECASE,
)

# Bảng cần farm_id filter (qua ZoneId)
_WAREHOUSE_TABLE_PATTERN = re.compile(
    r'"Tbl_Warehouse"',
    re.IGNORECASE,
)

# Pattern nhận biết farm_id đã được inject
_FARM_ID_PATTERN = re.compile(
    r'"ZoneId"\s*=\s*\'[0-9a-f\-]{36}\'',
    re.IGNORECASE,
)

# Pattern placeholder :farm_id chưa được thay thế
_PLACEHOLDER_PATTERN = re.compile(r":farm_id", re.IGNORECASE)

# UUID validation
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def validate_sql(sql: str, farm_id: str) -> str:
    """
    Validate và sanitize SQL trước khi execute.

    Args:
        sql:     SQL string do VannaAgent generate.
        farm_id: UUID của trang trại hiện tại.

    Returns:
        SQL đã được validate và inject farm_id (nếu cần).

    Raises:
        GuardrailError: Khi SQL vi phạm bất kỳ rule nào.
    """
    if not sql or not sql.strip():
        raise GuardrailError("SQL rỗng — không thể thực thi.")

    # 1. Validate farm_id là UUID hợp lệ
    _check_farm_id(farm_id)

    # 2. Chỉ cho phép SELECT
    _check_select_only(sql)

    # 3. Block SQL injection patterns
    _check_injection(sql)

    # 4. Inject / verify farm_id trong SQL
    sql = _inject_farm_id(sql, farm_id)

    # 5. Verify IsDeleted = false có mặt (warning nếu thiếu)
    _warn_missing_soft_delete(sql)

    logger.info("[Guardrail] SQL passed all checks")
    return sql


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def _check_farm_id(farm_id: str) -> None:
    """farm_id phải là UUID hợp lệ."""
    if not farm_id or not _UUID_PATTERN.match(farm_id.strip()):
        raise GuardrailError(
            f"farm_id không hợp lệ: '{farm_id}'. Phải là UUID dạng xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx."
        )


def _check_select_only(sql: str) -> None:
    """Chặn tất cả DML/DDL. SQL phải bắt đầu bằng SELECT hoặc WITH."""
    stripped = sql.strip().upper()

    # Phải bắt đầu bằng SELECT hoặc WITH (CTE)
    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        raise GuardrailError(
            "SQL phải bắt đầu bằng SELECT hoặc WITH. "
            f"Phát hiện: '{sql.strip()[:50]}...'"
        )

    # Kiểm tra các từ khóa bị cấm trong toàn bộ câu SQL
    match = _FORBIDDEN_KEYWORDS.search(sql)
    if match:
        raise GuardrailError(
            f"SQL chứa từ khóa bị cấm: '{match.group()}'. "
            "Chỉ cho phép SELECT."
        )


def _check_injection(sql: str) -> None:
    """Block SQL injection patterns phổ biến."""
    match = _INJECTION_PATTERNS.search(sql)
    if match:
        raise GuardrailError(
            f"SQL chứa pattern nguy hiểm: '{match.group()}'. Bị từ chối."
        )


def _inject_farm_id(sql: str, farm_id: str) -> str:
    """
    Đảm bảo farm_id được inject đúng cách.

    Logic:
    - Nếu SQL có placeholder ':farm_id' → replace bằng UUID thật.
    - Nếu SQL query Tbl_Warehouse nhưng KHÔNG có ZoneId filter → raise error.
    - Nếu SQL không dùng Tbl_Warehouse (VD: chỉ query Tbl_Material) → OK, không cần farm_id.
    - Nếu SQL có Tbl_CentralWarehouse → OK, không cần farm_id.
    """
    farm_id = farm_id.strip()

    # Replace placeholder :farm_id nếu có
    if _PLACEHOLDER_PATTERN.search(sql):
        sql = _PLACEHOLDER_PATTERN.sub(f"'{farm_id}'", sql)
        logger.debug("[Guardrail] replaced :farm_id placeholder → '%s'", farm_id)

    # Kiểm tra: nếu có Tbl_Warehouse thì phải có ZoneId filter
    uses_warehouse = _WAREHOUSE_TABLE_PATTERN.search(sql)
    if uses_warehouse:
        has_farm_filter = _FARM_ID_PATTERN.search(sql)
        if not has_farm_filter:
            raise GuardrailError(
                "SQL sử dụng Tbl_Warehouse nhưng THIẾU điều kiện ZoneId = '<farm_id>'. "
                "Vi phạm farm isolation rule."
            )

        # Cross-farm check: ZoneId phải khớp đúng farm_id hiện tại
        wrong_farm = re.search(
            r'"ZoneId"\s*=\s*\'([0-9a-f\-]{36})\'',
            sql,
            re.IGNORECASE,
        )
        if wrong_farm and wrong_farm.group(1).lower() != farm_id.lower():
            raise GuardrailError(
                f"Cross-farm query bị chặn! "
                f"SQL filter ZoneId='{wrong_farm.group(1)}' nhưng farm hiện tại='{farm_id}'."
            )

    return sql


def _warn_missing_soft_delete(sql: str) -> None:
    """
    Log warning nếu không thấy IsDeleted = false trong SQL.
    Không raise error vì một số CTE phức tạp có thể đặt ở sub-query.
    """
    if '"IsDeleted"' in sql and "false" not in sql.lower():
        logger.warning(
            "[Guardrail] WARNING: SQL có cột IsDeleted nhưng không thấy filter '= false'. "
            "Kiểm tra lại soft-delete condition."
        )
    elif '"IsDeleted"' not in sql:
        logger.warning(
            "[Guardrail] WARNING: SQL không chứa điều kiện IsDeleted. "
            "Có thể trả về dữ liệu đã xóa."
        )