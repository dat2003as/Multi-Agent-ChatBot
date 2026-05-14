from uuid import UUID
from datetime import datetime
from typing import Optional
import re
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from app.dependencies import get_db, get_memory, verify_api_key
from app.api.farm_context import resolve_authorized_farm_name
from app.orchestrator.orchestrator import Orchestrator
from app.debug.tracer import get_logger
from app.debug.request_tracker import tracker as debug_tracker, steps_var

log = get_logger(__name__)
router = APIRouter(
    prefix="/chat", 
    tags=["chat"],
    dependencies=[Depends(verify_api_key)]
)
_orchestrator = Orchestrator()

# UUID pattern: 8-4-4-4-12 hex digits
_UUID_PATTERN = re.compile(
    r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
    re.IGNORECASE
)

def _contains_uuid(text: str) -> bool:
    """Check if text contains UUID pattern. Used to block UUID-based queries for security."""
    return bool(_UUID_PATTERN.search(text))

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier", example="sess-12345")
    farm_id: Optional[UUID] = Field(None, description="Farm identifier", example="57f842cf-bc5d-42a8-8198-e0647fe2b05c")
    role: str = Field("farmer", description="User role: farmer, employee, manager, admin", example="admin")
    message: str = Field(..., min_length=1, max_length=2000, example="Giá tôm hôm nay thế nào?")

class ChatResponse(BaseModel):
    status: str = Field("success", example="success")
    message: str = Field("Response generated successfully", example="Response generated successfully")
    session_id: str = Field(..., example="sess-12345")
    answer: str = Field(..., example="Giá tôm sú loại 20 con/kg hôm nay là 240.000 đ/kg.")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db=Depends(get_db),
    memory=Depends(get_memory),
):
    # Khởi tạo tracking context cho request hiện tại
    steps_var.set([])
    debug_tracker.add_step("USER", "API_CHAT", "Received chat request")

    log.info("api.chat", session_id=req.session_id, farm_id=str(req.farm_id) if req.farm_id else None)

    # SECURITY: Block UUID-based queries to prevent data enumeration attacks
    if _contains_uuid(req.message):
        log.warning(
            "api.chat.uuid_query_blocked",
            session_id=req.session_id,
            message_preview=req.message[:100],
            reason="UUID patterns detected in user input"
        )
        return ChatResponse(
            session_id=req.session_id,
            answer="Hệ thống chưa cập nhật thông tin này! (UUID-based queries are not allowed)",
            status="error"
        )

    authorized_farm_name = None

    if req.farm_id:
        authorized_farm_name = await resolve_authorized_farm_name(req.farm_id)

        if not authorized_farm_name:
            log.warning(
                "api.chat.farm_name_unresolved.proceed",
                farm_id=str(req.farm_id),
                note="Proceeding with farm_id-only context because farm name not found",
            )

    farm_name = authorized_farm_name

    debug_tracker.add_step("API_CHAT", "ORCHESTRATOR", "Forwarding to Orchestrator")

    answer = await _orchestrator.run(
        user_message=req.message,
        session_id=req.session_id,
        farm_id=req.farm_id,
        role=req.role,
        db=db,
        memory=memory,
        farm_name=farm_name,
    )
    return ChatResponse(session_id=req.session_id, answer=answer)
