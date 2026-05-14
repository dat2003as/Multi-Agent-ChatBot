"""
Intent Detector — LLM-based Domain Routing (A1/A2/A3/A4)

Purpose:
  - Use LLM to analyze user question semantically
  - Route to correct Vanna agent (vanna_a1, vanna_a2, vanna_a3, vanna_a4)
  - Cache results for token efficiency
  - Fail-safe: return None if LLM unavailable

Domains:
  A1 — Farm Operations: aquaculture ponds, water quality, diseases, environment, aquatic life, farming
  A2 — Inventory Warehouse: inventory, stock levels, purchase/sale orders, materials, goods
  A3 — Report Analytics: costs, revenue, KPIs, reports, profit, statistics, performance
  A4 — IoT & Scales: IoT devices, sensors, scales, monitoring, data collection
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field
from async_lru import alru_cache

from app.core.llm_clients import get_chat_client

logger = logging.getLogger(__name__)

DomainType = Literal["a1", "a2", "a3", "a4"]


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
class DomainDetection(BaseModel):
    """LLM response schema for domain detection."""
    domain: DomainType | None = Field(
        None,
        description="Detected domain: a1 (Farm Ops), a2 (Inventory), a3 (Analytics), a4 (IoT), or null if out of scope"
    )
    reason: str = Field(
        "",
        description="Short explanation for the detection"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent domain router. Classify the user's question into one of 4 domains:

A1 — Farm Operations: aquaculture ponds, shrimp/fish farming, disease control, water environment, aquatic organisms, small-scale agriculture
A2 — Inventory Warehouse: inventory management, stock levels, purchase/sale orders, materials, goods, warehouse bins, central storage, suppliers
A3 — Report Analytics: costs, revenue, KPIs, reports, profit, statistics, performance metrics
A4 — IoT & Devices: IoT devices, sensors, scales, monitoring systems, data collection, device readings, device cabinet (tủ thiết bị), device hub, electronic cabinet (tủ điện), sensor readings, device status, weight measurement, device list, cameras

🔑 CRITICAL KEYWORDS FOR A4:
- "tủ" (cabinet), "tủ IoT" (IoT cabinet), "tủ thiết bị" (device cabinet), "tủ điện" (electrical cabinet)
- "DeviceHub", "device", "sensor", "cân" (scale), "cảm biến" (sensor), "camera"
- "bộ cảnh báo" (alarm device), "máy sục khí" (aerator), "máy sục khí Oxi" (oxygen aerator)
- "máy bơm" (pump), "quạt nước" (paddlewheel), "thiết bị đo" (measuring device)
- "lắp đặt" / "chưa lắp đặt" (installed/uninstalled), "đang hoạt động" (active), "bảo trì" (maintenance)
- Anything about device status, readings, monitoring, weight data, equipment count, installation status

DO NOT confuse aquaculture equipment/devices (A4) with inventory warehouse materials (A2).
"Bộ cảnh báo", "Máy sục khí", "Máy bơm", "Camera" are physical devices → A4, not inventory items.

Respond with JSON only: {"domain": "a1"|"a2"|"a3"|"a4"|null, "reason": "brief explanation"}

If the question does not belong to any of these 4 domains, return domain=null.
Analyze the context carefully and determine the true intent of the user.
DO NOT confuse "tủ thiết bị" (device cabinet) with inventory warehouse bins — "tủ" referring to IoT/device is A4, not A2."""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
@alru_cache(maxsize=1000)
async def detect_domain(
    question: str,
    llm_query: str | None = None,
) -> DomainType | None:
    """
    Analyze user question and detect which domain to route to.
    Uses LLM with semantic understanding. Results are cached.

    Args:
        question: Raw user message.
        llm_query: Optional LLM-generated query (preferred over raw question).
                   When available, this is prioritized since LLM has already clarified context.

    Returns:
        "a1", "a2", "a3", "a4" if intent is clearly detected.
        None if question is out of scope or LLM is unavailable (Fail-Safe).
    """
    if not question and not llm_query:
        logger.debug("[IntentDetector] no input provided")
        return None

    text_to_check = (llm_query or question).strip()
    if not text_to_check:
        logger.debug("[IntentDetector] input is empty after stripping")
        return None

    logger.debug(
        "[IntentDetector] analyzing question: %r%s",
        question[:100] if question else "",
        f" (llm_query: {llm_query[:80]})" if llm_query else ""
    )

    try:
        client = get_chat_client()
        
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this question:\n\n{text_to_check}"}
            ]
        )
        
        response_text = (response.choices[0].message.content or "{}").strip()
        detection = DomainDetection.model_validate_json(response_text)
        
        if detection.domain:
            logger.info(
                "[IntentDetector] question routed to domain=%s (reason: %s)",
                detection.domain.upper(),
                detection.reason[:50]
            )
        else:
            logger.info(
                "[IntentDetector] question is out of scope (reason: %s)",
                detection.reason[:50]
            )
        
        return detection.domain

    except json.JSONDecodeError as exc:
        logger.warning("[IntentDetector] JSON parsing failed: %s, returning None (Fail-Safe)", exc)
        return None
    except Exception as exc:
        logger.error("[IntentDetector] LLM error: %s, returning None (Fail-Safe)", exc)
        return None
