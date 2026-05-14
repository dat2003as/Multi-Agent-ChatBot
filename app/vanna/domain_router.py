"""
Warehouse Domain Router — Map farm_id → warehouse domain (A1/A2/A3/A4)

Schema instances on VANNA_DB_AZURE_URL:
  - vanna_a1: A1 — Nông dân quy mô nhỏ (Crop Operations)
  - vanna_a2: A2 — Kho vật tư trung tâm (Inventory Warehouse) 
  - vanna_a3: A3 — Quản lý nguồn chủ (Source Management)
  - vanna_a4: A4 — Thiết bị IOT và Cân (IoT & Scales)

Strategy (Smart Routing):
  1. Query BE_DB (Tbl_Farm.WarehouseCategory) để lấy domain của farm
  2. Validate domain có trong ACTIVE_DOMAINS
  3. Fallback → environment VANNA_DOMAIN config
  4. Fallback → DEFAULT_DOMAIN nếu tất cả không khả dụng
"""

from __future__ import annotations

import logging
import os
from typing import Literal
from uuid import UUID

import psycopg2
import psycopg2.extras

from app.core.config import settings

logger = logging.getLogger(__name__)

DomainType = Literal["a1", "a2", "a3", "a4"]

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_DOMAINS = {"a1", "a2", "a3", "a4"}  # All 4 domains now active
DEFAULT_DOMAIN = "a2"
VANNA_DOMAIN_ENV = os.getenv("VANNA_DOMAIN", DEFAULT_DOMAIN).lower()  # From environment
DOMAIN_NAMES = {
    "a1": "A1 — Nông dân quy mô nhỏ (Crop Operations)",
    "a2": "A2 — Kho vật tư trung tâm (Inventory Warehouse)",
    "a3": "A3 — Quản lý nguồn chủ (Source Management)",
    "a4": "A4 — Thiết bị IOT và Cân (IoT & Scales)",
}


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────
class DomainRouterError(Exception):
    """Domain routing failed."""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_farm_warehouse_domain(farm_id: UUID | str) -> DomainType:
    """
    Lấy warehouse domain cho farm từ BE DB.

    Strategy:
      1. Query BE DB — tìm bảng Farm/FarmConfig có column warehouse_domain
      2. Validate nó là A1/A2/A3/A4 có sẵn
      3. Fallback → DEFAULT_DOMAIN nếu:
         - Farm không tìm thấy
         - Column không tồn tại
         - Domain không có sẵn  (A3 coming soon)

    Args:
        farm_id: UUID của farm

    Returns:
        domain_key: "a1", "a2", "a3", hoặc "a4"
    """
    farm_id_str = str(farm_id)
    logger.info("[DomainRouter] resolving domain for farm_id=%s", farm_id_str)

    try:
        # BƯỚC 1: Query BE DB
        domain = _query_farm_domain_from_be_db(farm_id_str)

        if domain:
            domain_lower = domain.lower()
            if domain_lower in ACTIVE_DOMAINS:
                logger.info(
                    "[DomainRouter] farm_id=%s -> domain=%s ✅",
                    farm_id_str, domain_lower
                )
                return domain_lower
            else:
                logger.warning(
                    "[DomainRouter] farm_id=%s has domain=%s (not ready), "
                    "falling back to %s",
                    farm_id_str, domain_lower, DEFAULT_DOMAIN
                )
                return DEFAULT_DOMAIN

    except Exception as exc:
        logger.warning(
            "[DomainRouter] query failed for farm_id=%s: %s, fallback to %s",
            farm_id_str, exc, DEFAULT_DOMAIN
        )

    # BƯỚC 2: Fallback → Default (A2)
    logger.info(
        "[DomainRouter] farm_id=%s not found or error, using default domain=%s",
        farm_id_str, DEFAULT_DOMAIN
    )
    return DEFAULT_DOMAIN


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL
# ─────────────────────────────────────────────────────────────────────────────
def _query_farm_domain_from_be_db(farm_id: str) -> str | None:
    """
    Query BE DB để lấy warehouse_domain của farm.

    Tìm cột warehouse_domain trong bảng (tên có thể là):
      - Tbl_Farm.WarehouseCategory
      - Tbl_Farm.Domain
      - Tbl_FarmConfig.WarehouseDomain
      - ...

    Returns:
        domain string (ví dụ: "A2") hoặc None nếu không tìm thấy
    """
    conn = None
    try:
        # Connect tới BE DB
        conn = psycopg2.connect(settings.BE_DB_URL)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # SQL query — tìm domain của farm
        # Giả định: Tbl_Farm có cột WarehouseCategory (chứa "A1", "A2", "A3")
        sql = """
        SELECT "WarehouseCategory" 
        FROM "Tbl_Farm" 
        WHERE "FarmId" = %s 
        AND "IsDeleted" = false 
        LIMIT 1
        """

        cur.execute(sql, (farm_id,))
        row = cur.fetchone()
        cur.close()

        if row and row.get("WarehouseCategory"):
            return row["WarehouseCategory"]
        return None

    except psycopg2.Error as exc:
        logger.debug("[DomainRouter] SQL error: %s", exc)
        # Tiếp tục — fallback logic sẽ handle
        return None
    except Exception as exc:
        logger.debug("[DomainRouter] unexpected error querying BE DB: %s", exc)
        return None
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Print active domains (untuk debug)
# ─────────────────────────────────────────────────────────────────────────────
def print_active_domains():
    """Debug helper — in danh sách domain sẵn sàng."""
    print("\n" + "=" * 70)
    print("VANNA DOMAINS — ACTIVE STATUS")
    print("=" * 70)
    for domain_key in ["a1", "a2", "a3", "a4"]:
        status = "✅ READY" if domain_key in ACTIVE_DOMAINS else "⏳ COMING SOON"
        print(f"  {domain_key.upper():3s} — {DOMAIN_NAMES[domain_key]:50s} {status}")
    print(f"\n  Default: {DEFAULT_DOMAIN.upper()}")
    print("=" * 70 + "\n")
