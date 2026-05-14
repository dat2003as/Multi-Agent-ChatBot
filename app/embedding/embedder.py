from __future__ import annotations
import tiktoken
from typing import Sequence
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError
from app.core.config import settings
from app.debug.tracer import get_logger

log = get_logger(__name__)

EMBED_MAX_TOKENS = 8191
EMBED_BATCH_SIZE = 16

class Embedder:
    def __init__(self) -> None:
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_EMBED_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMBED_API_KEY or settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_EMBED_API_VERSION or settings.AZURE_OPENAI_API_VERSION,
        )
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._deployment = settings.AZURE_OPENAI_EMBED_DEPLOYMENT

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def truncate_to_budget(self, text: str, max_tokens: int = EMBED_MAX_TOKENS) -> str:
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated = self._enc.decode(tokens[:max_tokens])
        log.warning("embed.truncated", original_tokens=len(tokens), max_tokens=max_tokens)
        return truncated

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._deployment,
            input=texts,
            dimensions=settings.EMBEDDING_DIM,
        )
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str) -> list[float]:
        safe_text = self.truncate_to_budget(text)
        results = await self._embed_batch([safe_text])
        log.debug("embed.one", tokens=self.count_tokens(safe_text))
        return results[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        safe_texts = [self.truncate_to_budget(t) for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(safe_texts), EMBED_BATCH_SIZE):
            batch = safe_texts[i : i + EMBED_BATCH_SIZE]
            log.debug("embed.batch", batch_index=i, batch_size=len(batch))
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        log.info("embed.many.done", total=len(all_embeddings))
        return all_embeddings
