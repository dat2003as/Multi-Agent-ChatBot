import logging
import sys
import structlog
from app.core.config import settings

def configure_logging():
    # Cấu hình logging tiêu chuẩn của Python để structlog có thể sử dụng
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.LOG_LEVEL),
    )

    # Suppress noisy HTTP library logs (chỉ hiện WARNING trở lên)
    for noisy_logger in (
        "httpcore", "httpcore.http11", "httpcore.connection",
        "httpx", "openai", "openai._base_client",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
            if settings.LOG_LEVEL == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    return structlog.get_logger(name)

def trace_llm_call(iteration: int, finish_reason: str, tool_calls: list, prompt_tokens: int, completion_tokens: int):
    if not settings.TRACE_TOOL_CALLS:
        return
    log = get_logger("llm.trace")
    log.debug(
        "llm.call",
        iteration=iteration,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

def trace_tool_call(tool: str, **kwargs):
    if not settings.TRACE_TOOL_CALLS:
        return
    log = get_logger("tool.trace")
    log.debug("tool.executed", tool=tool, **kwargs)
