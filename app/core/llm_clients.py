from __future__ import annotations

from functools import lru_cache

from openai import AsyncAzureOpenAI

from app.core.config import settings


@lru_cache(maxsize=1)
def get_chat_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )
