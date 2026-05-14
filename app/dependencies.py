from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from redis.asyncio.cluster import RedisCluster
from app.core.config import settings
from app.memory.session_memory import SessionMemory
import logging

log = logging.getLogger(__name__)

# Database engine and sessionmaker
connect_args = {}
if settings.DATABASE_SSL_MODE in ["require", "verify-ca", "verify-full"]:
    connect_args["ssl"] = True

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    connect_args=connect_args
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with async_session() as session:
        yield session

# Redis Cluster Client - Direct initialization
async def _create_redis_client():
    """Create RedisCluster client for Azure Redis and standard Redis."""
    try:
        client = RedisCluster.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        await client.ping()
        log.info("[Redis] Connected to Redis using RedisCluster client")
        return client
    except Exception as exc:
        log.error(f"[Redis] Failed to connect: {exc}")
        raise RuntimeError(f"Failed to connect to Redis: {exc}")

# Global Redis client instance
redis_client: RedisCluster | None = None

async def _init_redis():
    """Initialize Redis client on app startup."""
    global redis_client
    if redis_client is None:
        redis_client = await _create_redis_client()
        log.info("[Redis] Redis client initialized on startup")

async def _close_redis():
    """Close Redis client on app shutdown."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        log.info("[Redis] Redis client closed on shutdown")

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify that the provided API Key matches the configured one."""
    if api_key == settings.API_KEY:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )

async def get_memory():
    """Get session memory with Redis backend."""
    global redis_client
    if redis_client is None:
        await _init_redis()
    return SessionMemory(redis_client)
