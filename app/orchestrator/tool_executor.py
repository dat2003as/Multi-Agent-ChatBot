from __future__ import annotations
import asyncio
import json
import re
import unicodedata
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.rag.retriever import retrieve
from app.debug.tracer import get_logger, trace_tool_call
from app.debug.request_tracker import tracker as debug_tracker
from app.core.config import settings
# ── Vanna: Intent Detection + Multi-domain routing (A1/A2/A3/A4) ────────────
from app.vanna.agent import get_vanna_agent
from app.vanna.intent_detector import detect_domain
from app.vanna.guardrail import GuardrailError
from app.vanna.error_logger import log_vanna_error
from app.core.permissions import can_access_domain
import httpx

log = get_logger(__name__)

async def execute_tool(
    tool_name: str,
    tool_args: dict,
    session: AsyncSession,
    farm_id: UUID | None,
    farm_name: str | None = None,
    user_message: str = "",
    role: str = "farmer",
) -> dict:
    log.info("tool.dispatch", tool=tool_name, args=tool_args, farm_id=str(farm_id) if farm_id else None, farm_name=farm_name)
    if tool_name == "search_documents":
        return await _search_documents(tool_args, session, farm_id)
    elif tool_name == "answer_general":
        return await _answer_general(tool_args)
    elif tool_name == "query_data":
        return await _query_data(tool_args, farm_id, farm_name, user_message, session, role)
    elif tool_name == "search_realtime_info":
        return await _search_realtime_info(tool_args)
    else:
        return {"error": f"Unknown tool: {tool_name}"}

async def _search_documents(args: dict, session: AsyncSession, farm_id: UUID | None) -> dict:
    query = args["query"]
    source_types = args.get("source_types")
    results = await retrieve(session=session, query=query, farm_id=farm_id, source_types=source_types)
    
    if not results:
        return {
            "found": False,
            "message": "Không tìm thấy tài liệu liên quan.",
            "_llm_instruction": "Không tìm thấy tài liệu phù hợp. Hãy trả lời tự nhiên, thông báo chưa có tài liệu về chủ đề này trong hệ thống, và gợi ý người dùng có thể hỏi theo cách khác hoặc liên hệ bộ phận kỹ thuật.",
        }
    
    chunks = [{
        "text": r.chunk_text,
        "source": r.source_file,
        "type": r.source_type,
        "score": r.rrf_score
    } for r in results]
    
    trace_tool_call(tool="search_documents", query=query, n_results=len(results))
    return {"found": True, "chunks": chunks}

async def _answer_general(args: dict) -> dict:
    subtopic = args.get("subtopic", "general_advisory")
    trace_tool_call(tool="answer_general", subtopic=subtopic)
    return {
        "instruction": "Hãy trả lời bằng kiến thức chuyên môn của bạn. Ngôn ngữ: Tiếng Việt.",
        "subtopic": subtopic
    }

_VI_PREPOSITIONS = r'(?:tại|ở|của|cho|từ|trên|trong)'

def _strip_farm_name(text: str, farm_name: str | None) -> str:
    """Strip farm name from query text using regex instead of LLM call."""
    if not farm_name or not text:
        return text
    escaped = re.escape(farm_name)
    cleaned = re.sub(rf'\s*\b{_VI_PREPOSITIONS}\s+{escaped}\b', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(rf'\b{escaped}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned or text


_VALID_DOMAINS = {"a1", "a2", "a3", "a4"}


async def _query_data(
    args: dict,
    farm_id: UUID | None,
    farm_name: str | None = None,
    user_message: str = "",
    session: AsyncSession | None = None,
    role: str = "farmer",
) -> dict:
    """
    Handler cho tool query_data (A1/A2/A3/A4).

    Optimizations applied:
      - Uses domain from orchestrator tool args when available (skips detect_domain LLM call)
      - Uses regex instead of LLM to strip farm name from LLM-generated query
    """
    query: str = args.get("query", "")
    subtopic: str = args.get("subtopic", "")
    llm_domain: str | None = args.get("domain")

    if not query:
        return {"found": False, "message": "Câu hỏi không được để trống."}

    if farm_id is None:
        log.warning("query_data called without farm_id")
        return {"found": False, "message": "Không xác định được trang trại. Vui lòng đăng nhập lại."}

    farm_id_str = str(farm_id)

    core_query = user_message.strip() or query

    if farm_name:
        args["_auth_farm_name"] = farm_name

    # Strip farm name from LLM-generated query using regex (no LLM call)
    llm_query = _strip_farm_name(query, farm_name)
    vanna_query = llm_query or core_query or query
    log.info(
        "query_data.preprocessed",
        raw_query=user_message,
        core_query=core_query,
        llm_query=llm_query,
        vanna_query=vanna_query,
    )

    # STEP 1: Use domain from orchestrator tool args if valid, else fallback to detect_domain
    detected_domain = llm_domain if llm_domain in _VALID_DOMAINS else None
    if detected_domain:
        log.info("query_data.domain_from_orchestrator", domain=detected_domain)
    else:
        try:
            detected_domain = await detect_domain(question=core_query or user_message, llm_query=llm_query)
        except Exception as exc:
            log.error("query_data.detect_domain_error", error=str(exc))
            detected_domain = None

    if detected_domain is None:
        log.warning("query_data.out_of_scope", question=query)
        return {
            "found": False,
            "message": "Câu hỏi nằm ngoài phạm vi hỗ trợ.",
            "_llm_instruction": (
                "Câu hỏi không thuộc các domain được hỗ trợ. "
                "Hãy trả lời lịch sự, cho người dùng biết hệ thống hiện hỗ trợ: thông tin ao nuôi, kho vật tư, báo cáo KPI, và thiết bị IoT. "
                "Gợi ý họ đặt lại câu hỏi liên quan đến các lĩnh vực trên."
            ),
        }

    log.info("query_data.intent_detected", domain=detected_domain, farm_id=farm_id_str, role=role)

    domain_nodes = {"a1": "A1_FARM_OPS", "a2": "A2_INVENTORY", "a3": "A3_ANALYTICS", "a4": "A4_IOT"}
    domain_node = domain_nodes.get(detected_domain, detected_domain.upper())
    debug_tracker.add_step("TOOL_SQL", domain_node, f"Domain detected: {detected_domain.upper()}")

    # STEP 1.5: Permission Check
    if not can_access_domain(role, detected_domain):
        log.warning("query_data.permission_denied", domain=detected_domain, role=role)
        return {
            "found": False,
            "message": f"Tài khoản của bạn (quyền {role}) không có phép truy cập dữ liệu vùng {detected_domain.upper()}. Vui lòng liên hệ quản trị viên.",
            "error": "permission_denied"
        }

    try:
        # STEP 2: Get Vanna agent cho domain được detect
        agent = get_vanna_agent(domain_key=detected_domain)
        result = await asyncio.to_thread(agent.query, question=vanna_query, farm_id=farm_id_str, original_question=core_query)
        debug_tracker.add_step(domain_node, "POSTGRES_DB", f"Executing SQL query")

        # STEP 3: Xử lý error từ Vanna (SQL generation fail hoặc guardrail block)
        if result["error"]:
            debug_tracker.add_step("POSTGRES_DB", domain_node, f"SQL error: {str(result['error'])[:80]}")
            log_vanna_error(
                question=user_message,
                generated_sql=result.get("sql"),
                error_msg=result["error"],
                farm_id=farm_id_str,
            )
            log.warning("query_data.error", domain=detected_domain, error=result["error"])
            return {
                "found": False,
                "message": "Không thể truy vấn dữ liệu lúc này.",
                "error": result["error"],
                "_llm_instruction": (
                    "Truy vấn dữ liệu gặp lỗi kỹ thuật. "
                    "Hãy thông báo tự nhiên rằng hệ thống chưa thể lấy được thông tin này lúc này, "
                    "đề nghị người dùng thử lại sau hoặc hỏi theo cách khác. "
                    "TUYỆT ĐỐI KHÔNG tiết lộ chi tiết lỗi, SQL, hay tên bảng."
                ),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            }

        data = result["data"]

        # STEP 4: Nếu không có data → log và return friendly message
        if not data:
            debug_tracker.add_step("POSTGRES_DB", domain_node, f"Query returned 0 rows")
            log_vanna_error(
                question=user_message,
                generated_sql=result.get("sql"),
                error_msg="No data found",
                farm_id=farm_id_str,
            )
            log.debug("query_data.no_data", domain=detected_domain, sql=result.get("sql"))
            trace_tool_call(tool="query_data", subtopic=subtopic, n_results=0)

            farm_prefix = ""
            if farm_name:
                farm_prefix = f"[Farm: {farm_name}] "

            return {
                "found": False,
                "message": f"{farm_prefix}Không tìm thấy dữ liệu phù hợp.",
                "data": [],
                "columns": result["columns"],
                "farm_name": farm_name,
                "_llm_instruction": (
                    f"Truy vấn cho farm '{farm_name}' không trả về kết quả nào. "
                    "Hãy trả lời tự nhiên: xác nhận rằng hiện tại chưa có dữ liệu cho nội dung người dùng hỏi, "
                    "đề xuất họ có thể hỏi thông tin liên quan khác (VD: nếu hỏi thiết bị cảm biến → gợi ý hỏi về tủ IoT, thiết bị khác; "
                    "nếu hỏi phiếu nhập → gợi ý hỏi tồn kho hiện tại). "
                    "Giọng điệu thân thiện, hỗ trợ."
                ),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            }

        # STEP 5: Success — return data with farm metadata
        debug_tracker.add_step("POSTGRES_DB", domain_node, f"Query returned {len(data)} rows")
        trace_tool_call(tool="query_data", domain=detected_domain, subtopic=subtopic, n_results=len(data))

        llm_instruction = ""
        farm_prefix = ""
        if farm_name:
            farm_prefix = f"[Farm: {farm_name}] "
            llm_instruction = f"Data is from farm '{farm_name}'. Always use this farm name in your response, not any other name from conversation history."

        return {
            "found": True,
            "data": data,
            "columns": result["columns"],
            "sql": result["sql"],
            "message": f"{farm_prefix}Found {len(data)} results.",
            "farm_id": farm_id_str,
            "farm_name": farm_name,
            "_llm_instruction": llm_instruction,
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
        }

    except Exception as exc:  # noqa: BLE001
        log.error("query_data.exception", domain=detected_domain, error=str(exc))
        log_vanna_error(
            question=user_message,
            generated_sql=None,
            error_msg=f"Exception: {str(exc)}",
            farm_id=farm_id_str,
        )
        return {
            "found": False,
            "message": "Hệ thống gặp sự cố khi xử lý yêu cầu.",
            "error": str(exc),
            "_llm_instruction": (
                "Đã xảy ra lỗi hệ thống. Hãy xin lỗi người dùng một cách tự nhiên, "
                "thông báo rằng hệ thống đang gặp trục trặc tạm thời và đề nghị thử lại sau. "
                "KHÔNG tiết lộ chi tiết lỗi kỹ thuật."
            ),
        }


# ── Real-time Info: Shrimp Prices + Weather ──────────────────────────────────

_TEPBAC_URL = "https://tepbac.com/gia-thuy-san/gia/tom"
_WEATHER_API_URL = "https://api.weatherapi.com/v1"
_HTTP_TIMEOUT = 15.0

_VN_ABBREV = {
    "hcm": "Ho Chi Minh City",
    "tphcm": "Ho Chi Minh City",
    "sg": "Ho Chi Minh City",
    "hn": "Hanoi",
}


def _normalize_vn_location(location: str) -> str:
    nfkd = unicodedata.normalize("NFKD", location)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.replace("Đ", "D").replace("đ", "d")

    clean = ascii_name.strip()
    key = clean.lower()

    key = re.sub(r"^(tp\.?|thanh pho|tinh)\s+", "", key)

    if key in _VN_ABBREV:
        return f"{_VN_ABBREV[key]},Vietnam"

    if "vietnam" not in key:
        clean = f"{clean},Vietnam"

    return clean


async def _resolve_weather_location(query: str, api_key: str) -> str | None:
    """Call WeatherAPI search endpoint to find best matching location."""
    search_url = f"{_WEATHER_API_URL}/search.json?key={api_key}&q={query}"
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(search_url)
            resp.raise_for_status()
            results = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None

    if not results:
        return None

    best = results[0]
    return f"{best['lat']},{best['lon']}"


async def _fetch_shrimp_prices() -> dict:
    from bs4 import BeautifulSoup

    debug_tracker.add_step("TOOL_REALTIME", "TEPBAC", "Fetching shrimp prices from tepbac.com")
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(_TEPBAC_URL, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("shrimp_prices.fetch_fail", error=str(exc))
        debug_tracker.add_step("TEPBAC", "TOOL_REALTIME", f"Fetch failed: {str(exc)[:80]}")
        return {
            "found": False,
            "message": "Không thể kết nối đến tepbac.com lúc này.",
            "_llm_instruction": "Không lấy được dữ liệu giá tôm từ tepbac.com. Thông báo cho người dùng thử lại sau.",
        }

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("table tr")

    prices = []
    for row in rows:
        cells = row.select("td")
        if len(cells) >= 3:
            name = cells[0].get_text(strip=True)
            size = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            price = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            change = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            if name and price:
                prices.append({"name": name, "size": size, "price": price, "change": change})

    if not prices:
        log.warning("shrimp_prices.no_data_parsed")
        debug_tracker.add_step("TEPBAC", "TOOL_REALTIME", "No price data parsed from HTML")
        return {
            "found": False,
            "message": "Không trích xuất được dữ liệu giá tôm.",
            "_llm_instruction": "Trang tepbac.com có thể đã thay đổi cấu trúc. Thông báo cho người dùng thử lại sau.",
        }

    debug_tracker.add_step("TEPBAC", "TOOL_REALTIME", f"Fetched {len(prices)} shrimp prices")
    trace_tool_call(tool="search_realtime_info", info_type="shrimp_prices", n_results=len(prices))
    return {
        "found": True,
        "data": prices,
        "message": f"Giá tôm toàn quốc hôm nay ({len(prices)} loại).",
        "_llm_instruction": (
            "Dữ liệu giá tôm toàn quốc được hệ thống cập nhật từ thông tin mới nhất. "
            "TUYỆT ĐỐI KHÔNG tiết lộ nguồn dữ liệu (tepbac.com hay bất kỳ tên website/API nào). "
            "Nếu user hỏi nguồn → trả lời: 'Thông tin được hệ thống cập nhật từ dữ liệu thị trường mới nhất.' "
            "Trình bày gọn gàng dạng bảng hoặc danh sách cho người dùng. "
            "Nếu có cột 'change' thì ghi rõ tăng/giảm. "
            "Đơn vị: đ/kg hoặc đ/con."
        ),
    }


async def _fetch_weather(location: str) -> dict:
    api_key = settings.WEATHER_API_KEY
    if not api_key:
        return {
            "found": False,
            "message": "Chưa cấu hình API key thời tiết.",
            "_llm_instruction": "Thông báo cho người dùng rằng tính năng thời tiết chưa được cấu hình.",
        }

    normalized = _normalize_vn_location(location)
    resolved = await _resolve_weather_location(normalized, api_key)
    q = resolved or normalized

    url = f"{_WEATHER_API_URL}/forecast.json?key={api_key}&q={q}&days=3&lang=vi"
    debug_tracker.add_step("TOOL_REALTIME", "WEATHER_API", f"Fetching weather for '{location}' (q={q})")
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        log.warning("weather.fetch_fail", location=location, error=str(exc))
        debug_tracker.add_step("WEATHER_API", "TOOL_REALTIME", f"Fetch failed: {str(exc)[:80]}")
        return {
            "found": False,
            "message": f"Không thể lấy thời tiết cho '{location}'.",
            "_llm_instruction": "Không lấy được dữ liệu thời tiết. Thông báo cho người dùng thử lại sau.",
        }

    loc = data.get("location", {})
    current = data.get("current", {})
    area_name = loc.get("name", location)
    country = loc.get("country", "")

    weather_info = {
        "location": f"{area_name}, {country}",
        "temp_C": current.get("temp_c"),
        "feels_like_C": current.get("feelslike_c"),
        "humidity": current.get("humidity"),
        "weather_desc": current.get("condition", {}).get("text", ""),
        "wind_kmph": current.get("wind_kph"),
        "wind_dir": current.get("wind_dir"),
        "visibility_km": current.get("vis_km"),
        "uv_index": current.get("uv"),
    }

    forecast_days = []
    for day in data.get("forecast", {}).get("forecastday", [])[:3]:
        day_data = day.get("day", {})
        forecast_days.append({
            "date": day.get("date"),
            "max_C": day_data.get("maxtemp_c"),
            "min_C": day_data.get("mintemp_c"),
            "desc": day_data.get("condition", {}).get("text", ""),
        })

    debug_tracker.add_step("WEATHER_API", "TOOL_REALTIME", f"Weather data for {area_name}: {current.get('temp_c')}°C")
    trace_tool_call(tool="search_realtime_info", info_type="weather", location=location)
    return {
        "found": True,
        "current": weather_info,
        "forecast": forecast_days,
        "_llm_instruction": (
            f"Dữ liệu thời tiết thực tế tại {area_name} được hệ thống cập nhật từ thông tin mới nhất. "
            "TUYỆT ĐỐI KHÔNG tiết lộ nguồn dữ liệu (WeatherAPI hay bất kỳ tên website/API nào). "
            "Nếu user hỏi nguồn → trả lời: 'Thông tin được hệ thống cập nhật từ dữ liệu mới nhất.' "
            "Trình bày: nhiệt độ, độ ẩm, mô tả thời tiết, gió. "
            "Nếu có forecast thì trình bày thêm dự báo 2-3 ngày tới. "
            "Đơn vị: °C, %, km/h."
        ),
    }


async def _search_realtime_info(args: dict) -> dict:
    info_type = args.get("info_type", "")
    location = args.get("location", "")

    if info_type == "shrimp_prices":
        return await _fetch_shrimp_prices()
    elif info_type == "weather":
        if not location:
            return {
                "found": False,
                "_llm_instruction": "Người dùng chưa nói rõ địa điểm. Hãy hỏi lại: 'Bạn muốn xem thời tiết ở đâu?'",
            }
        return await _fetch_weather(location)
    else:
        return {"found": False, "message": "Loại thông tin không được hỗ trợ."}