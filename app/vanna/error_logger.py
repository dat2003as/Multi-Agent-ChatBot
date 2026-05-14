"""
app/vanna/error_logger.py
=========================
Ghi lỗi Vanna SQL vào CSV để review & retrain định kỳ.

Theo diagram: Error Capture → CSV Logger (vanna_error_log.csv)
Fields: timestamp, farm_id, question, generated_sql, error_log
"""
from __future__ import annotations

import csv
import logging
import os
import uuid
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_LOG_PATH = os.getenv("VANNA_ERROR_LOG_PATH", "vanna_error_log.csv")
_FIELDS   = ["timestamp", "farm_id", "question", "generated_sql", "error_log"]


def log_vanna_error(
    question: str,
    error_msg: str,
    farm_id: str = "",
    generated_sql: str | None = None,
) -> None:
    """Append một dòng lỗi vào CSV. Tự tạo file + header nếu chưa có."""
    try:
        file_exists = os.path.isfile(_LOG_PATH)
        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
                "farm_id":       farm_id,
                "question":      question,
                "generated_sql": generated_sql or "",
                "error_log":     error_msg,
            })
    except Exception as exc:  # noqa: BLE001
        log.error("vanna_error_logger failed: %s", exc)