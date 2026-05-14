from datetime import datetime
from typing import List, Dict, Any, Optional
import contextvars
import time
from pydantic import BaseModel

class RequestLog(BaseModel):
    id: int
    session_id: Optional[str] = None
    farm_id: Optional[str] = None
    role: Optional[str] = None
    query: str
    input_type: str
    output_type: str
    tokens: Dict[str, Any]
    llm_components: str
    tool_used: str
    intent: str
    message_output: str
    duration_ms: Optional[int] = None
    status: str = "success"
    error_message: Optional[str] = None
    created_at: str
    steps: List[Dict[str, Any]] = []

steps_var: contextvars.ContextVar[List[Dict[str, Any]]] = contextvars.ContextVar("steps")

class RequestTracker:
    def __init__(self, max_logs: int = 100):
        self._logs: List[RequestLog] = []
        self._max_logs = max_logs
        self._counter = 0

    def add_log(
        self,
        query: str,
        input_type: str,
        output_type: str,
        tokens: Dict[str, Any],
        llm_components: str,
        tool_used: str,
        intent: str,
        message_output: str,
        session_id: Optional[str] = None,
        farm_id: Optional[str] = None,
        role: Optional[str] = None,
        duration_ms: Optional[int] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ):
        steps = steps_var.get(None)
        if steps is None:
            steps = []
            
        self._counter += 1
        log_entry = RequestLog(
            id=self._counter,
            session_id=session_id,
            farm_id=farm_id,
            role=role,
            query=query,
            input_type=input_type,
            output_type=output_type,
            tokens=tokens,
            llm_components=llm_components,
            tool_used=tool_used,
            intent=intent,
            message_output=message_output,
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
            created_at=datetime.now().strftime("%H:%M:%S"),
            steps=steps
        )
        self._logs.append(log_entry)  # Add to end (1, 2, 3...)
        if len(self._logs) > self._max_logs:
            self._logs.pop(0) # Remove oldest

    def clear_logs(self):
        self._logs = []
        self._counter = 0

    def get_logs(self) -> List[RequestLog]:
        return self._logs

    def add_step(self, from_node: str, to_node: str, action: str):
        steps = steps_var.get(None)
        if steps is None:
            steps = []
            steps_var.set(steps)
        
        # We will assume that at the beginning of the request, `steps_var.set([])` is explicitly called.
        steps.append({
            "from": from_node,
            "to": to_node,
            "action": action,
            "timestamp": time.time()
        })

# Singleton instance
tracker = RequestTracker()
