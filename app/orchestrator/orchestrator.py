from __future__ import annotations
import json
import re
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from openai import BadRequestError
from app.core.config import settings
from app.core.llm_clients import get_chat_client
from app.orchestrator.tool_registry import TOOLS
from app.orchestrator.tool_executor import execute_tool
from app.prompts.system_prompt import build_system_prompt
from app.memory.session_memory import SessionMemory
from app.debug.tracer import get_logger, trace_llm_call
from app.debug.request_tracker import tracker as debug_tracker
import time
import traceback

log = get_logger(__name__)
MAX_TOOL_ITERATIONS = 5


_QUESTION_ENDINGS = re.compile(
    r"(?:làm\s+(?:thế\s+nào|sao)|thế\s+nào|như\s+thế\s+nào|cách\s+nào)\s*\??\s*$",
    re.IGNORECASE,
)
_ACTION_KEYWORDS = re.compile(
    r"\b(?:muốn\s+)?(?:xóa|xoá|thêm|sửa|cập nhật|tạo|chỉnh sửa)\b",
    re.IGNORECASE,
)
_CODING_KEYWORDS = re.compile(
    r"\b(?:python|java|javascript|react|fastapi|html|css|sql|api|oop|code|lập trình)\b",
    re.IGNORECASE,
)

def _is_app_how_to_query(message: str) -> bool:
    """Detect 'Tôi muốn [action]...thì làm thế nào?' pattern that LLM misclassifies as DB write."""
    msg = message.strip()
    if _CODING_KEYWORDS.search(msg):
        return False
    return bool(_ACTION_KEYWORDS.search(msg) and _QUESTION_ENDINGS.search(msg))
1
_MAX_LLM_ROWS = 300


def _compact_for_llm(result: dict) -> dict:
    """Convert data array-of-dicts to columns+rows format for token efficiency."""
    data = result.get("data")
    if not data or not isinstance(data, list):
        return result
    if not data[0] or not isinstance(data[0], dict):
        return result

    columns = result.get("columns") or list(data[0].keys())
    total_rows = len(data)
    display = data[:_MAX_LLM_ROWS]

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        return str(v)

    rows = [[_safe(row.get(c)) for c in columns] for row in display]

    compact = {k: v for k, v in result.items() if k not in ("data", "sql", "prompt_tokens", "completion_tokens")}
    compact["columns"] = columns
    compact["rows"] = rows
    compact["total_rows"] = total_rows
    if total_rows > _MAX_LLM_ROWS:
        compact["note"] = f"Showing top {_MAX_LLM_ROWS} of {total_rows} rows."
    return compact


class Orchestrator:
    def __init__(self) -> None:
        self._client = get_chat_client()

    async def run(
        self,
        user_message: str,
        session_id: str,
        farm_id: UUID | None,
        db: AsyncSession,
        memory: SessionMemory,
        role: str = "farmer",
        farm_name: str | None = None,
    ) -> str:
        history = await memory.get(session_id)
        system_prompt = build_system_prompt(farm_id=farm_id, farm_name=farm_name, role=role)
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        tools_metadata = []
        components = ["LLM"]

        start_time = time.time()
        final_answer = "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý."
        status = "success"
        error_message = None

        try:
            force_search = _is_app_how_to_query(user_message)
            if force_search:
                log.info("orchestrator.force_search_documents", query=user_message[:80])

            for i in range(MAX_TOOL_ITERATIONS):
                debug_tracker.add_step("ORCHESTRATOR", "LLM", f"Iteration {i}: Calling LLM")

                tool_choice = "auto"
                if i == 0 and force_search:
                    tool_choice = {"type": "function", "function": {"name": "search_documents"}}

                response = await self._client.chat.completions.create(
                    model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                    temperature=0.2,
                )
                debug_tracker.add_step("LLM", "ORCHESTRATOR", "LLM returned response")
                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                
                # Tracking tokens
                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                
                trace_llm_call(iteration=i, finish_reason=finish_reason, 
                               tool_calls=[tc.function.name for tc in (msg.tool_calls or [])],
                               prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                               completion_tokens=response.usage.completion_tokens if response.usage else 0)

                if not msg.tool_calls:
                    final_answer = msg.content or ""
                    await memory.append(session_id, "user", user_message)
                    await memory.append(session_id, "assistant", final_answer)
                    return final_answer

                messages.append(msg)
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tools_metadata.append(tool_name)
                    
                    tool_node_map = {
                        "query_data": "TOOL_SQL",
                        "search_documents": "TOOL_RAG",
                        "search_realtime_info": "TOOL_REALTIME",
                        "answer_general": "TOOL_GENERAL",
                    }
                    tool_node = tool_node_map.get(tool_name, "TOOL_GENERAL")
                    debug_tracker.add_step("ORCHESTRATOR", tool_node, f"Calling tool {tool_name}")

                    if tool_name == "search_documents":
                        components.extend(["Embedding", "Vector DB"])

                    result = await execute_tool(tool_name,
                                                json.loads(tool_call.function.arguments),
                                                db, farm_id, farm_name, user_message, role)
                    
                    debug_tracker.add_step(tool_node, "ORCHESTRATOR", f"Tool {tool_name} returned result")
                    
                    if isinstance(result, dict):
                        total_prompt_tokens += result.get("prompt_tokens", 0)
                        total_completion_tokens += result.get("completion_tokens", 0)

                    llm_payload = _compact_for_llm(result) if isinstance(result, dict) else result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(llm_payload, ensure_ascii=False)
                    })
                    
                    # Extract zone/warehouse name from result and add hint to LLM
                    if isinstance(result, dict):
                        # Ưu tiên: result có column name → query trả về
                        zone_name = result.get("zone_name") or result.get("Tên trại") or result.get("Tên kho")
                        
                        # Fallback: lấy từ args nếu tool_executor validate rồi store
                        if not zone_name:
                            zone_name = result.get("_auth_farm_name")
                        
                        # Fallback 2: Query DB trực tiếp nếu vẫn không có (ensure hint luôn có)
                        if not zone_name and farm_id:
                            try:
                                from sqlalchemy import text
                                db_result = await db.execute(
                                    text('SELECT "Name" FROM "Tbl_Zone" WHERE "Id" = :farm_id'),
                                    {"farm_id": str(farm_id)}
                                )
                                row = db_result.fetchone()
                                zone_name = row[0] if row else None
                            except Exception:
                                pass
                        
                        if zone_name:
                            messages.append({
                                "role": "user",
                                "content": f"[Lưu ý kỹ: Tên trại/kho chính xác từ database là \"{zone_name}\". Hãy sử dụng tên này thay vì tên người dùng cung cấp trong câu trả lời]"
                            })
            
            # If loop finishes without returning, it exceeded MAX_TOOL_ITERATIONS
            status = "error"
            error_message = "Exceeded max tool iterations"
            return final_answer

        except BadRequestError as e:
            error_body = str(e)
            if "content_filter" in error_body or "ResponsibleAIPolicyViolation" in error_body:
                log.warning("orchestrator.content_filter_blocked", error=error_body[:200])
                final_answer = "Yêu cầu không hợp lệ."
                status = "error"
                error_message = "content_filter_blocked"
            else:
                status = "error"
                error_message = str(e)
                log.error(f"Error in orchestrator run: {e}\n{traceback.format_exc()}")
            return final_answer

        except Exception as e:
            status = "error"
            error_message = str(e)
            log.error(f"Error in orchestrator run: {e}\n{traceback.format_exc()}")
            return final_answer

        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            debug_tracker.add_step("ORCHESTRATOR", "API_CHAT", "Flow completed")
            # Log to Tracker
            debug_tracker.add_log(
                query=user_message,
                input_type="system + user" if not history else "system + history + user",
                output_type="final answer" if not tools_metadata else "tool call + final answer",
                tokens={
                    "total": total_prompt_tokens + total_completion_tokens,
                    "prompt": total_prompt_tokens,
                    "completion": total_completion_tokens,
                    "llm": f"{total_prompt_tokens} + {total_completion_tokens}",
                    "emb": "N/A"
                },
                llm_components=", ".join(list(set(components))),
                tool_used=", ".join(list(set(tools_metadata))) if tools_metadata else "none",
                intent=tools_metadata[0] if tools_metadata else "out_of_scope",
                message_output=final_answer,
                session_id=session_id,
                farm_id=str(farm_id) if farm_id else None,
                role=role,
                duration_ms=duration_ms,
                status=status,
                error_message=error_message
            )
