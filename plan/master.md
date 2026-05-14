# AI Chatbot — Quy trình RAG & Tổng quát

**Kế hoạch Triển khai Thực tế**
Tác giả: Hưng Huy Cao | Phiên bản: 1.0 | Công nghệ: Azure OpenAI GPT-4.1-mini · pgvector · FastAPI

---

## Mục lục

1. [Tổng quan Kiến trúc](#1-tong-quan-kien-truc)
2. [Cấu trúc Dự án](#2-cau-truc-du-an)
3. [Môi trường & Thư viện](#3-moi-truong--thu-vien)
4. [Schema Database — pgvector](#4-schema-database--pgvector)
5. [Dịch vụ Embedding](#5-dich-vu-embedding)
6. [Luồng nạp Tài liệu](#6-luong-nap-tai-lieu)
7. [Dịch vụ Truy xuất — RAG](#7-dich-vu-truy-xuat--rag)
8. [GPT-4o Orchestrator — Quyết định Tool](#8-gpt-4o-orchestrator--quyet-dinh-tool)
9. [Câu lệnh gợi ý (Prompts)](#9-cau-lenh-goi-y-prompts)
10. [Tầng API — FastAPI](#10-tang-api--fastapi)
11. [Bộ nhớ ngắn hạn](#11-bo-nho-ngan-han)
12. [Gỡ lỗi & Giám sát](#12-go-loi--giam-sat)
13. [Kiểm thử](#13-kiem-thu)
14. [Danh sách Kiểm tra Triển khai](#14-danh-sach-kiem-tra-trien-khai)

---

## 1. Tổng quan Kiến trúc

```
Tin nhắn Người dùng
     │
     ▼
┌──────────────────────────────────────────────┐
│  L2 — Tầng Xác thực (FE/BE)                   │
│  farm_id được lấy từ JWT, chuyển tới AI       │
└──────────────┬───────────────────────────────┘
               │ farm_id, user_message
               ▼
┌──────────────────────────────────────────────┐
│  L3 — Bộ nhớ ngắn hạn                         │
│  Redis: lưu 3 lượt hội thoại gần nhất         │
└──────────────┬───────────────────────────────┘
               │ tin nhắn + lịch sử
               ▼
┌──────────────────────────────────────────────┐  ← PHẠM VI CỦA KẾ HOẠCH NÀY
│  L4 — GPT-4o Agentic Orchestrator            │
│  Azure OpenAI · Function Calling             │
│                                              │
│  Các Tool được đăng ký:                       │
│   • search_documents(query, farm_id, k)      │
│   • answer_general(query)                    │
│   [execute_sql — không bao gồm trong đây]    │
└──────┬────────────────────┬──────────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐    ┌────────────────────┐
│  Công cụ RAG │    │  Công cụ LLM Chung │
│  pgvector   │    │  Suy luận trực tiếp │
│  hybrid     │    │  không DB / vector  │
│  search     │    └────────────────────┘
└──────┬──────┘
       │ top-k chunks
       ▼
┌──────────────────────────────────────────────┐
│  Azure PostgreSQL 17.7 + pgvector            │
│  Bảng doc_embeddings                         │
└──────────────────────────────────────────────┘
       │
       ▼
GPT-4o tổng hợp câu trả lời cuối cùng → Phản hồi L8
```

**Phạm vi của tài liệu này**
- Công cụ `search_documents`: RAG trên các tài liệu đã tải lên (SOP, FAQ, hướng dẫn).
- Công cụ `answer_general`: Trả lời trực tiếp từ LLM cho tư vấn nông nghiệp, trò chuyện phiếm.
- `execute_sql` (Vanna): **Loại trừ** — triển khai riêng biệt.

---

## 2. Cấu trúc Dự án

```
chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Điểm khởi đầu FastAPI
│   ├── config.py                  # Cấu hình (pydantic-settings)
│   ├── dependencies.py            # DI: DB pool, Redis, embedder
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py                # Endpoint POST /chat
│   │   └── ingest.py              # Endpoint POST /ingest
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # Vòng lặp tool-calling của GPT-4o
│   │   ├── tool_registry.py       # Định nghĩa Tool gửi tới OpenAI
│   │   └── tool_executor.py       # Điều phối gọi Tool → services
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py           # Tìm kiếm hybrid với pgvector
│   │   ├── reranker.py            # Tùy chọn: rerank với cross-encoder
│   │   └── chunker.py             # Chia nhỏ tài liệu (chunks)
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py            # Azure OpenAI text-embedding-3-large
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── pipeline.py            # Điều phối: load → chunk → embed → store
│   │   └── loaders.py             # Bộ tải file PDF / DOCX / MD
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── session_memory.py      # Bộ nhớ ngắn hạn Redis
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system_prompt.py       # Câu lệnh hệ thống chính
│   │   └── rag_prompt.py          # Câu lệnh tổng hợp RAG
│   │
│   └── debug/
│       ├── __init__.py
│       └── tracer.py              # Ghi log gỡ lỗi có cấu trúc
│
├── db/
│   └── migrations/
│       └── 001_create_doc_embeddings.sql
│
├── scripts/
│   ├── ingest_docs.py             # CLI: python scripts/ingest_docs.py --path ./docs
│   └── test_retrieval.py          # CLI: kiểm tra chất lượng tìm kiếm
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_orchestrator.py
│   └── integration/
│       └── test_chat_flow.py
│
├── docs/                          # Tài liệu mẫu để nạp vào hệ thống
├── .env.example
├── .env                           # Không bao giờ commit file này
├── requirements.txt
├── docker-compose.yml             # Local dev: Postgres + Redis
└── README.md
```

---

## 3. Môi trường & Thư viện

### `.env.example`

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large

# PostgreSQL + pgvector
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/chatbot
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis (Bộ nhớ ngắn hạn)
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=1800

# RAG
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.35
RAG_HYBRID_ALPHA=0.7        # 0 = thuần BM25, 1 = thuần vector
EMBEDDING_DIM=3072           # text-embedding-3-large

# Debug
LOG_LEVEL=DEBUG              # DEBUG | INFO | WARNING | ERROR
TRACE_TOOL_CALLS=true
TRACE_RETRIEVAL=true
```

### `requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
pydantic-settings==2.3.0
openai==1.35.3
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.30
pgvector==0.3.1
redis[asyncio]==5.0.6
pypdf2==3.0.1
python-docx==1.1.2
tiktoken==0.7.0
tenacity==8.3.0
structlog==24.2.0
python-multipart==0.0.9
httpx==0.27.0
pytest==8.2.2
pytest-asyncio==0.23.7
```

---

## 4. Schema Database — pgvector

### `db/migrations/001_create_doc_embeddings.sql`

```sql
-- Kích hoạt pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng lưu trữ embeddings của tài liệu
CREATE TABLE doc_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id         UUID,                           -- NULL = tài liệu chung (SOP, FAQ)
    source_file     TEXT NOT NULL,                  -- tên file gốc
    source_type     TEXT NOT NULL,                  -- 'sop' | 'faq' | 'guide' | 'manual'
    chunk_index     INTEGER NOT NULL,               -- vị trí trong file nguồn
    chunk_text      TEXT NOT NULL,                  -- nội dung thô của đoạn này
    chunk_tokens    INTEGER,                        -- số lượng token để quản lý chi phí
    embedding       vector(3072),                   -- text-embedding-3-large
    metadata        JSONB DEFAULT '{}',             -- số trang, chương, tags, v.v.
    is_deleted      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Chỉ mục tương đồng Vector (HNSW cho >1M dòng)
CREATE INDEX idx_doc_embed_vector
    ON doc_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Chỉ mục lọc cho tìm kiếm hybrid
CREATE INDEX idx_doc_embed_farm      ON doc_embeddings (farm_id);
CREATE INDEX idx_doc_embed_type      ON doc_embeddings (source_type);
CREATE INDEX idx_doc_embed_deleted   ON doc_embeddings (is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_doc_embed_metadata  ON doc_embeddings USING GIN (metadata);

-- Chỉ mục tìm kiếm toàn văn Full-text search (cho thành phần từ khóa BM25)
CREATE INDEX idx_doc_embed_fts
    ON doc_embeddings
    USING GIN (to_tsvector('vietnamese', chunk_text));

-- Trigger cập nhật thời gian
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_doc_embeddings_updated_at
    BEFORE UPDATE ON doc_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

---

## 5. Dịch vụ Embedding

### `app/embedding/embedder.py`

```python
"""
Dịch vụ Azure OpenAI Embedding
- Model: text-embedding-3-large (dim=3072)
- Có cơ chế thử lại (retry) với exponential backoff
- Kiểm soát hạn mức token trước khi gọi API
"""

from __future__ import annotations

import asyncio
import tiktoken
from typing import Sequence

from openai import AsyncAzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APITimeoutError

from app.config import settings
from app.debug.tracer import get_logger

log = get_logger(__name__)

# Giới hạn token tối đa cho model
EMBED_MAX_TOKENS = 8191
EMBED_BATCH_SIZE = 16   # Kích thước batch khuyến nghị của Azure


class Embedder:
    def __init__(self) -> None:
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
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
        log.warning(
            "embed.truncated",
            original_tokens=len(tokens),
            max_tokens=max_tokens,
        )
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
        )
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str) -> list[float]:
        safe_text = self.truncate_to_budget(text)
        results = await self._embed_batch([safe_text])
        log.debug("embed.one", tokens=self.count_tokens(safe_text))
        return results[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embedding danh sách văn bản theo từng batch."""
        safe_texts = [self.truncate_to_budget(t) for t in texts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(safe_texts), EMBED_BATCH_SIZE):
            batch = safe_texts[i : i + EMBED_BATCH_SIZE]
            log.debug("embed.batch", batch_index=i, batch_size=len(batch))
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        log.info("embed.many.done", total=len(all_embeddings))
        return all_embeddings
```

---

## 6. Luồng nạp Tài liệu

### `app/ingest/loaders.py`

```python
"""
Bộ tải file: PDF, DOCX, Markdown → văn bản thuần
"""

import io
from pathlib import Path

import PyPDF2
import docx as python_docx

from app.debug.tracer import get_logger

log = get_logger(__name__)


def load_pdf(path: Path) -> str:
    reader = PyPDF2.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
        log.debug("loader.pdf.page", page=i, chars=len(text))
    full_text = "\n\n".join(pages)
    log.info("loader.pdf.done", path=str(path), total_chars=len(full_text))
    return full_text


def load_docx(path: Path) -> str:
    doc = python_docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    log.info("loader.docx.done", path=str(path), paragraphs=len(paragraphs))
    return full_text


def load_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    log.info("loader.md.done", path=str(path), chars=len(text))
    return text


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    loaders = {".pdf": load_pdf, ".docx": load_docx, ".md": load_markdown, ".txt": load_markdown}
    if suffix not in loaders:
        raise ValueError(f"Định dạng file không hỗ trợ: {suffix}")
    return loaders[suffix](path)
```

### `app/rag/chunker.py`

```python
"""
Chia đoạn ngữ nghĩa (Semantic chunking) có gối đầu (overlap).
Chiến lược: chia theo đoạn văn → gộp các đoạn nhỏ → giới hạn cứng theo ngân sách token.
"""

from __future__ import annotations

import re
import tiktoken

from app.config import settings
from app.debug.tracer import get_logger

log = get_logger(__name__)

CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
CHUNK_MAX_TOKENS = 512


class Chunker:
    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_paragraphs(self, text: str) -> list[str]:
        # Tách theo dòng trống đôi hoặc tiêu đề markdown
        parts = re.split(r"\n{2,}|(?=^#{1,3} )", text, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]

    def chunk(self, text: str, source_file: str = "") -> list[dict]:
        """
        Trả về danh sách các {chunk_index, chunk_text, chunk_tokens}.
        """
        paragraphs = self._split_paragraphs(text)
        chunks: list[dict] = []
        current_parts: list[str] = []
        current_tokens = 0
        overlap_buffer = ""

        for para in paragraphs:
            para_tokens = self._count(para)

            # Đoạn quá dài → tách cứng
            if para_tokens > CHUNK_MAX_TOKENS:
                if current_parts:
                    self._flush(chunks, current_parts, overlap_buffer)
                    overlap_buffer = self._make_overlap(current_parts)
                    current_parts = []
                    current_tokens = 0
                for hard_chunk in self._hard_split(para):
                    self._flush(chunks, [hard_chunk], overlap_buffer)
                    overlap_buffer = self._make_overlap([hard_chunk])
                continue

            if current_tokens + para_tokens > CHUNK_TARGET_TOKENS and current_parts:
                self._flush(chunks, current_parts, overlap_buffer)
                overlap_buffer = self._make_overlap(current_parts)
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            self._flush(chunks, current_parts, overlap_buffer)

        log.info(
            "chunker.done",
            source=source_file,
            total_chunks=len(chunks),
            avg_tokens=int(sum(c["chunk_tokens"] for c in chunks) / max(len(chunks), 1)),
        )
        return chunks

    def _flush(self, chunks: list, parts: list[str], overlap: str) -> None:
        text = (overlap + "\n\n" + "\n\n".join(parts)).strip()
        chunks.append(
            {
                "chunk_index": len(chunks),
                "chunk_text": text,
                "chunk_tokens": self._count(text),
            }
        )

    def _make_overlap(self, parts: list[str]) -> str:
        """Lấy CHUNK_OVERLAP_TOKENS token cuối cùng từ chunk hiện tại."""
        combined = " ".join(parts)
        tokens = self._enc.encode(combined)
        overlap_tokens = tokens[-CHUNK_OVERLAP_TOKENS:]
        return self._enc.decode(overlap_tokens)

    def _hard_split(self, text: str) -> list[str]:
        tokens = self._enc.encode(text)
        result = []
        for i in range(0, len(tokens), CHUNK_TARGET_TOKENS):
            result.append(self._enc.decode(tokens[i : i + CHUNK_TARGET_TOKENS]))
        return result
```

---

## 7. Dịch vụ Truy xuất — RAG

### `app/rag/retriever.py`

```python
"""
Tìm kiếm Hybrid: Tương đồng Vector Cosine + Full-text search PostgreSQL (BM25-style).
Gộp kết quả qua Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from uuid import UUID
from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding.embedder import Embedder
from app.config import settings
from app.debug.tracer import get_logger

log = get_logger(__name__)
embedder = Embedder()


class RetrievalResult:
    __slots__ = ("chunk_id", "chunk_text", "source_file", "source_type",
                 "chunk_index", "similarity_score", "fts_rank", "rrf_score", "metadata")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


async def retrieve(
    session: AsyncSession,
    query: str,
    farm_id: UUID | None = None,
    k: int | None = None,
    source_types: Sequence[str] | None = None,
) -> list[RetrievalResult]:
    """
    Hãy tìm kiếm hybrid và trả về k đoạn văn bản hàng đầu.
    """
    k = k or settings.RAG_TOP_K
    alpha = settings.RAG_HYBRID_ALPHA

    log.debug(
        "retrieval.start",
        query=query[:80],
        farm_id=str(farm_id),
        k=k,
        alpha=alpha,
        source_types=source_types,
    )

    # ── 1. Embedding câu truy vấn ──────────────────────────────────────────
    query_embedding = await embedder.embed_one(query)

    # ── 2. Bộ lọc theo trang trại và loại tài liệu ──────────────────────────
    type_clause = ""
    params: dict = {
        "embedding": query_embedding,
        "farm_id": str(farm_id) if farm_id else None,
        "limit": k * 3,        # lấy dư để gộp RRF
        "ts_query": " | ".join(query.split()),
    }
    if source_types:
        type_clause = "AND source_type = ANY(:source_types)"
        params["source_types"] = list(source_types)

    # ── 3. Tìm kiếm Vector ──────────────────────────────────────────────────
    vector_sql = sa.text(f"""
        SELECT
            id::text           AS chunk_id,
            chunk_text,
            source_file,
            source_type,
            chunk_index,
            1 - (embedding <=> :embedding::vector)  AS similarity_score,
            metadata
        FROM doc_embeddings
        WHERE is_deleted = FALSE
          AND (farm_id = :farm_id::uuid OR farm_id IS NULL)
          {type_clause}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :limit
    """)

    vec_rows = (await session.execute(vector_sql, params)).mappings().all()

    # ── 4. Tìm kiếm Toàn văn (Full-text search) ─────────────────────────────
    fts_sql = sa.text(f"""
        SELECT
            id::text           AS chunk_id,
            chunk_text,
            source_file,
            source_type,
            chunk_index,
            ts_rank_cd(
                to_tsvector('simple', chunk_text),
                to_tsquery('simple', :ts_query)
            )                  AS fts_rank,
            metadata
        FROM doc_embeddings
        WHERE is_deleted = FALSE
          AND (farm_id = :farm_id::uuid OR farm_id IS NULL)
          {type_clause}
          AND to_tsvector('simple', chunk_text) @@ to_tsquery('simple', :ts_query)
        ORDER BY fts_rank DESC
        LIMIT :limit
    """)

    fts_rows = (await session.execute(fts_sql, params)).mappings().all()

    # ── 5. Reciprocal Rank Fusion (Gộp hạng nghịch đảo) ─────────────────────
    RRF_K = 60
    scores: dict[str, dict] = {}

    for rank, row in enumerate(vec_rows):
        cid = row["chunk_id"]
        scores.setdefault(cid, {"row": row, "rrf": 0.0})
        scores[cid]["rrf"] += alpha * (1.0 / (RRF_K + rank + 1))

    for rank, row in enumerate(fts_rows):
        cid = row["chunk_id"]
        scores.setdefault(cid, {"row": row, "rrf": 0.0})
        scores[cid]["rrf"] += (1.0 - alpha) * (1.0 / (RRF_K + rank + 1))

    sorted_chunks = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)[:k]

    # ── 6. Lọc theo ngưỡng tương đồng ──────────────────────────────────────
    results = []
    for item in sorted_chunks:
        row = item["row"]
        sim = row.get("similarity_score") or 0.0
        if sim < settings.RAG_SIMILARITY_THRESHOLD:
            continue
        results.append(
            RetrievalResult(
                chunk_id=row["chunk_id"],
                chunk_text=row["chunk_text"],
                source_file=row["source_file"],
                source_type=row["source_type"],
                chunk_index=row["chunk_index"],
                similarity_score=round(float(sim), 4),
                fts_rank=float(row.get("fts_rank") or 0),
                rrf_score=round(item["rrf"], 6),
                metadata=dict(row["metadata"] or {}),
            )
        )

    log.info("retrieval.done", query=query[:80], returned=len(results))
    return results
```

---

## 8. GPT-4o Orchestrator — Quyết định Tool

### `app/orchestrator/tool_registry.py`

```python
"""
Định nghĩa hàm OpenAI cho việc chọn tool của GPT-4o.
Đây là các schema GPT-4o nhận để quyết định gọi tool nào.
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Tìm kiếm trong cơ sở tri thức tài liệu nội bộ về quy trình trang trại (SOP), "
                "hướng dẫn ứng dụng, FAQ kỹ thuật, điều trị bệnh, hướng dẫn thiết bị. "
                "Dùng khi người dùng hỏi LÀM THẾ NÀO, hỏi về quy trình, hướng dẫn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu truy vấn tìm kiếm chính xác bằng tiếng Việt.",
                    },
                    "source_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["sop", "faq", "guide", "manual"]},
                        "description": "Tùy chọn: lọc theo loại tài liệu.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_general",
            "description": (
                "Trả lời kiến thức chung về thủy sản - triệu chứng bệnh, môi trường nước, "
                "chiến lược cho ăn, quản lý ao nuôi, trò chuyện phiếm, chào hỏi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu hỏi cần trả lời bằng kiến thức chuyên gia.",
                    },
                    "subtopic": {
                        "type": "string",
                        "enum": ["disease_treatment", "water_quality", "feeding", "chitchat"],
                    },
                },
                "required": ["query"],
            },
        },
    },
]
```

---

## 9. Câu lệnh gợi ý (Prompts)

### `app/prompts/system_prompt.py`

```python
def build_system_prompt(farm_id: UUID | None = None) -> str:
    return f"""Bạn là trợ lý AI chuyên nghiệp — nền tảng quản lý trang trại thủy sản thông minh.

## Vai trò
Bạn hỗ trợ chủ trang trại, kỹ sư nuôi trồng giải quyết các vấn đề vận hành hàng ngày qua tiếng Việt.

## Thông tin phiên làm việc
Farm_id: {farm_id if farm_id else 'Chưa xác định'}

## Nguyên tắc chọn công cụ:
1. `search_documents`: Gọi khi cần tra cứu SOP, hướng dẫn sử dụng, hoặc quy trình kỹ thuật nội bộ.
2. `answer_general`: Gọi khi tư vấn kiến thức nuôi trồng chung hoặc trò chuyện.

## Định dạng câu trả lời:
- Luôn thân thiện, chuyên nghiệp bằng tiếng Việt.
- Nếu trích dẫn từ tài liệu: ghi rõ (Nguồn: ten_file.pdf) ở cuối.
- Nếu không biết: thừa nhận và đề xuất liên hệ hỗ trợ kỹ thuật.
"""
```

---

## 10. Tầng API — FastAPI

### `app/api/chat.py`

```python
@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db=Depends(get_db),
    memory=Depends(get_memory),
):
    log.info("api.chat", session_id=req.session_id)
    answer = await _orchestrator.run(
        user_message=req.message,
        session_id=req.session_id,
        farm_id=req.farm_id,
        db=db,
        memory=memory,
    )
    return ChatResponse(session_id=req.session_id, answer=answer)
```

---

## 11. Bộ nhớ ngắn hạn

Sử dụng **Redis** để lưu trữ 3-5 lượt hội thoại gần nhất nhằm duy trì ngữ cảnh câu hỏi của người dùng (ví dụ: "Nó dùng thế nào?" sau khi hỏi về một loại máy móc).

---

## 12. Gỡ lỗi & Giám sát

Sử dụng `structlog` để ghi log các bước:
- `llm.call`: Theo dõi số lượng token và kết quả chọn tool.
- `retrieval.done`: Theo dõi số đoạn văn bản tìm thấy và điểm số cao nhất.
- `tool.executed`: Theo dõi thời gian thực hiện các công cụ.

---

## 13. Kiểm thử

- **Unit test**: Kiểm tra bộ chia nhỏ văn bản (`Chunker`) đảm bảo không mất thông tin.
- **Integration test**: Chạy luồng Chat giả lập để kiểm tra RAG có lấy đúng tài liệu hay không.

---

## 14. Danh sách Kiểm tra Triển khai

- [ ] Azure PostgreSQL: Kích hoạt extension `vector`.
- [ ] Azure OpenAI: Đã tạo deployment cho GPT-4o và Embedding-v3.
- [ ] Redis: Kiểm tra kết nối và thời gian sống (TTL).
- [ ] Chạy script `python scripts/ingest_docs.py` để nạp dữ liệu ban đầu.
- [ ] Kiểm tra giá thành: Dự kiến **~$8.43 cho mỗi 1,000 tin nhắn**.

---

*AI Chatbot — Quy trình RAG v1.0*
*Tác giả: Hưng Huy Cao | Azure OpenAI GPT-4o + pgvector + FastAPI*