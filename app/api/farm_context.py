from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import psycopg2
import psycopg2.extras
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db_pool import get_be_connection, return_be_connection
from app.core.llm_clients import get_chat_client
from app.debug.tracer import get_logger

log = get_logger(__name__)


MISMATCH_MESSAGE = "Bạn đang truy cập dữ liệu không khớp với trang trại hiện tại."


def _normalize_for_match(value: str) -> str:
    """
    Normalize farm name for string comparison.
    
    Steps:
    1. Remove Vietnamese diacritics (NFD)
    2. Remove farm-type prefixes: "trại", "trai", "farm", "khu" (word boundaries)
    3. Convert to lowercase
    4. Clean whitespace (strip + collapse multiple spaces)
    
    Examples:
    - "Trại BE" -> "be"
    - "trại QC Kiên Giang" -> "qc kien giang"
    - "Farm Số 1" -> "so 1"
    """
    import unicodedata
    import re
    
    # Step 1: Remove Vietnamese diacritics
    normalized = unicodedata.normalize("NFD", value)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    
    # Step 2: Remove farm-type prefixes using word boundaries (\b)
    # Removes: trại, trai, farm, khu (case-insensitive)
    normalized = re.sub(r"\b(trại|trai|farm|khu)\b", "", normalized, flags=re.IGNORECASE)
    
    # Step 3: Convert to lowercase
    normalized = normalized.lower()
    
    # Step 4: Clean whitespace
    normalized = " ".join(normalized.strip().split())
    
    return normalized


@dataclass(frozen=True)
class FarmQueryFilterResult:
    blocked: bool
    message: str | None
    mentioned_farm: str | None
    core_query: str
    is_generic_reference: bool = False
    analyzer_prompt_tokens: int = 0
    analyzer_completion_tokens: int = 0


class FarmQueryExtraction(BaseModel):
    core_query: str = Field(
        description=(
            "Ý định cốt lõi của câu hỏi, ĐÃ ĐƯỢC XÓA BỎ hoàn toàn các từ chỉ tên trại riêng biệt. "
            "Giữ nguyên nếu là câu hỏi phiếm chỉ."
        )
    )
    mentioned_farm_name: Optional[str] = Field(
        default=None,
        description=(
            "Tên cụ thể của trang trại nếu được nhắc đến (VD: QC Kiên Giang, Trại BE). "
            "Bỏ qua chữ 'trại/farm/khu'. Trả về null nếu user không nhắc đích danh tên nào."
        ),
    )
    is_generic_reference: bool = Field(
        description=(
            "True nếu người dùng dùng đại từ phiếm chỉ (VD: 'trại này', 'khu này', 'chỗ đó', 'tên là gì', 'ở đây')."
        )
    )


def normalize_farm_name(value: str) -> str:
    return _normalize_for_match(value)


def _normalize_extraction_output(raw_query: str, extraction: FarmQueryExtraction) -> FarmQueryExtraction:
    """
    Normalize LLM extraction output.
    
    - Keep mentioned_farm_name as-is (will be normalized during comparison)
    - Remove diacritics from core_query if needed
    - Handle generic references
    """
    core_query = (extraction.core_query or "").strip() or raw_query.strip()
    mentioned = extraction.mentioned_farm_name.strip() if extraction.mentioned_farm_name else None

    if extraction.is_generic_reference:
        mentioned = None

    return FarmQueryExtraction(
        core_query=core_query,
        mentioned_farm_name=mentioned,
        is_generic_reference=extraction.is_generic_reference,
    )


async def extract_question_with_llm(raw_query: str) -> tuple[FarmQueryExtraction, int, int]:
    system_prompt = (
        "You are a query analyzer for an agricultural farm management system. "
        "Your task is to extract structured information from user questions and return valid JSON matching the FarmQueryExtraction schema. "
        "Provide only the JSON output without any explanation or markdown formatting. "
        "Analyze each user question and extract three fields: core_query, mentioned_farm_name, and is_generic_reference.\n"
        "\n"
        "CRITICAL RULES - DEVICE VS FARM DISTINCTION:\n"
        "- NEVER confuse specific equipment or infrastructure names with farm names. Equipment includes: physical devices (Cabinet 20, Electrical Cabinet, Pump, Sensor), water features (Pond A1, Basin B2), storage facilities (Warehouse W1). "
        "- If a user asks ONLY about a specific device, machine, or water feature WITHOUT explicitly mentioning a farm/zone name, you MUST set mentioned_farm_name to null. "
        "- Use mentioned_farm_name ONLY for actual farm or zone names. All other details belong in core_query.\n"
        "\n"
        "CRITICAL RULES - LOCATION VS FARM DISTINCTION:\n"
        "- When the user asks about weather, market prices, or shrimp prices for a province/city (e.g. 'thời tiết Cà Mau', 'giá tôm Sóc Trăng'), "
        "the province/city name is a LOCATION, NOT a farm name. Set mentioned_farm_name to null.\n"
        "- mentioned_farm_name is ONLY for farm/zone names in the context of querying farm operational data (e.g. 'trại BE', 'Farm QC Kiên Giang', 'khu nuôi số 1').\n"
        "- IMPORTANT: For weather/price queries, KEEP the location name in core_query. "
        "Example: 'thời tiết Cà Mau' → core_query='thời tiết Cà Mau', mentioned_farm_name=null."
    )

    user_prompt = (
        "Analyze the following question and return valid JSON:\n"
        f"Question: {raw_query}"
    )

    try:
        client = get_chat_client()
        response = await client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        p_tokens = response.usage.prompt_tokens if response.usage else 0
        c_tokens = response.usage.completion_tokens if response.usage else 0
        
        content = (response.choices[0].message.content or "{}").strip()
        payload = json.loads(content)
        extraction = FarmQueryExtraction.model_validate(payload)
        normalized = _normalize_extraction_output(raw_query, extraction)
        return normalized, p_tokens, c_tokens
    except (ValidationError, json.JSONDecodeError, IndexError, KeyError) as exc:
        log.warning("extract_question_with_llm.parse_fail", error=str(exc), query=raw_query)
    except Exception as exc:
        log.warning("extract_question_with_llm.fail", error=str(exc), query=raw_query)

    return FarmQueryExtraction(
        core_query=raw_query.strip(),
        mentioned_farm_name=None,
        is_generic_reference=True,
    ), 0, 0


async def preprocess_farm_query(raw_query: str, auth_farm_name: str | None) -> FarmQueryFilterResult:
    extraction, p_tokens, c_tokens = await extract_question_with_llm(raw_query)
    mentioned_farm = extraction.mentioned_farm_name
    core_query = raw_query.strip()

    if not mentioned_farm:
        return FarmQueryFilterResult(
            blocked=False,
            message=None,
            mentioned_farm=None,
            core_query=extraction.core_query or core_query,
            is_generic_reference=extraction.is_generic_reference,
            analyzer_prompt_tokens=p_tokens,
            analyzer_completion_tokens=c_tokens,
        )

    if not auth_farm_name:
        return FarmQueryFilterResult(
            blocked=True,
            message=MISMATCH_MESSAGE,
            mentioned_farm=mentioned_farm,
            core_query=core_query,
            is_generic_reference=extraction.is_generic_reference,
            analyzer_prompt_tokens=p_tokens,
            analyzer_completion_tokens=c_tokens,
        )

    mentioned_clean = normalize_farm_name(mentioned_farm)
    authorized_clean = normalize_farm_name(auth_farm_name)
    if mentioned_clean != authorized_clean:
        return FarmQueryFilterResult(
            blocked=True,
            message=MISMATCH_MESSAGE,
            mentioned_farm=mentioned_farm,
            core_query=core_query,
            is_generic_reference=extraction.is_generic_reference,
            analyzer_prompt_tokens=p_tokens,
            analyzer_completion_tokens=c_tokens,
        )

    cleaned_query = extraction.core_query.strip() if extraction.core_query else ""
    if not cleaned_query:
        cleaned_query = raw_query.strip()

    return FarmQueryFilterResult(
        blocked=False,
        message=None,
        mentioned_farm=mentioned_farm,
        core_query=cleaned_query,
        is_generic_reference=extraction.is_generic_reference,
        analyzer_prompt_tokens=p_tokens,
        analyzer_completion_tokens=c_tokens,
    )


async def _rollback_if_needed(db: AsyncSession, farm_id: str, stage: str) -> None:
    try:
        await db.rollback()
        log.info("resolve_farm_name.rollback", farm_id=farm_id, stage=stage)
    except Exception as rollback_error:
        log.warning(
            "resolve_farm_name.rollback.fail",
            farm_id=farm_id,
            stage=stage,
            error=str(rollback_error),
        )


async def resolve_authorized_farm_name(farm_id: UUID) -> str | None:
    """
    Resolve the authorized farm name from BE database by farm_id.
    Uses Tbl_Zone."Name" as the single source of truth.
    """
    farm_id_str = str(farm_id)
    conn = None

    try:
        conn = get_be_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cur.execute(
            'SELECT z."Name" FROM public."Tbl_Zone" z WHERE z."Id" = %s AND z."IsDeleted" = false LIMIT 1',
            (farm_id_str,),
        )
        row = cur.fetchone()
        cur.close()

        if row and row.get("Name"):
            log.info("resolve_farm_name.ok", farm_id=farm_id_str, farm_name=row["Name"])
            return str(row["Name"])

    except psycopg2.Error as e:
        log.warning("resolve_farm_name.fail", farm_id=farm_id_str, error=str(e), pgerror=getattr(e, 'pgerror', None))
    except Exception as e:
        log.warning("resolve_farm_name.error", farm_id=farm_id_str, error=str(e))
    finally:
        return_be_connection(conn)

    log.warning("resolve_farm_name.not_found", farm_id=farm_id_str)
    return None
