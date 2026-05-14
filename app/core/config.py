from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ─── Azure OpenAI — CHAT (GPT-4.1-mini) ───────────────────────────────────
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = "gpt-4.1-mini"


    AZURE_OPENAI_EMBED_DEPLOYMENT: str = "text-embedding-3-small"

    # ─── PostgreSQL + pgvector ─────────────────────────────────────────────────
    # Azure OpenAI Embedding (Optional, falls back to general if not set)
    AZURE_OPENAI_EMBED_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_EMBED_API_KEY: Optional[str] = None
    AZURE_OPENAI_EMBED_API_VERSION: Optional[str] = None

    # PostgreSQL + pgvector
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_SSL_MODE: str = "allow"  # allow, require, verify-ca, verify-full

    # ─── Redis ────────────────────────────────────────────────────────────────
    # Redis (For Azure: rediss://:<access_key>@<hostname>:6380/0)
    REDIS_URL: str 
    SESSION_TTL_SECONDS: int = 1800

    # ─── RAG ──────────────────────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.35
    RAG_HYBRID_ALPHA: float = 0.7
    EMBEDDING_DIM: int = 1536

    # ─── Vanna / PostgreSQL trực tiếp ─────────────────────────────────────────
    PG_HOST: str = ""
    PG_DB: str = ""
    PG_USER: str = ""
    PG_PASSWORD: str = ""
    PG_PORT: int = 5432
    
    BE_DB_URL: str = ""

    # ─── Security ─────────────────────────────────────────────────────────────
    API_KEY: str = ""

    # ─── Weather API ──────────────────────────────────────────────────────────
    WEATHER_API_KEY: str = ""

    # ─── Debug & Project ──────────────────────────────────────────────────────
    PROJECT_NAME: str = "Multi-Agent Chatbot"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "DEBUG"
    TRACE_TOOL_CALLS: bool = True
    TRACE_RETRIEVAL: bool = True

settings = Settings()