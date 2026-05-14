from __future__ import annotations
import json
from typing import Literal
from redis.asyncio.cluster import RedisCluster
from app.core.config import settings
from app.debug.tracer import get_logger

log = get_logger(__name__)
MAX_TURNS = 3

class SessionMemory:
    def __init__(self, redis: RedisCluster) -> None:
        """
        Initialize SessionMemory with Redis Cluster backend.
        
        Args:
            redis: RedisCluster client instance for Azure Redis Cluster.
        """
        self._redis = redis

    def _key(self, session_id: str) -> str:
        return f"chatbot:memory:{session_id}"

    async def get(self, session_id: str) -> list[dict]:
        """Get conversation history from Redis Cluster."""
        raw = await self._redis.get(self._key(session_id))
        if not raw: 
            return []
        return json.loads(raw)

    async def append(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        """Append a message to session history and maintain TTL."""
        messages = await self.get(session_id)
        messages.append({"role": role, "content": content})
        # Keep only last MAX_TURNS * 2 messages (latest turns)
        if len(messages) > MAX_TURNS * 2:
            messages = messages[-(MAX_TURNS * 2):]
        await self._redis.setex(
            self._key(session_id),
            settings.SESSION_TTL_SECONDS,
            json.dumps(messages, ensure_ascii=False)
        )

    async def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        await self._redis.delete(self._key(session_id))
