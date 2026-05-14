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
from app.core.config import settings
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
    k = k or settings.RAG_TOP_K
    alpha = settings.RAG_HYBRID_ALPHA

    log.debug("retrieval.start", query=query[:80], farm_id=str(farm_id))

    # 1. Embed query
    query_embedding_list = await embedder.embed_one(query)
    query_embedding = "[" + ",".join(map(str, query_embedding_list)) + "]"

    # 2. Build filters
    type_clause = ""
    params: dict = {
        "embedding": query_embedding,
        "farm_id": str(farm_id) if farm_id else None,
        "limit": k * 3,
        "ts_query": query,
    }
    if source_types:
        type_clause = "AND source_type = ANY(:source_types)"
        params["source_types"] = list(source_types)

    # 3. Vector search
    vector_sql = sa.text(f"""
        SELECT
            CAST(id AS text)   AS chunk_id,
            chunk_text,
            source_file,
            source_type,
            chunk_index,
            1 - (embedding OPERATOR(public.<=>) CAST(:embedding AS public.vector))  AS similarity_score,
            metadata
        FROM doc_embeddings
        WHERE is_deleted = FALSE
          AND (farm_id = CAST(:farm_id AS uuid) OR farm_id IS NULL)
          {type_clause}
        ORDER BY embedding OPERATOR(public.<=>) CAST(:embedding AS public.vector)
        LIMIT :limit
    """)
    vec_rows = (await session.execute(vector_sql, params)).mappings().all()

    # 4. Full-text search
    fts_sql = sa.text(f"""
        SELECT
            CAST(id AS text)   AS chunk_id,
            chunk_text,
            source_file,
            source_type,
            chunk_index,
            ts_rank_cd(
                to_tsvector('simple', chunk_text),
                websearch_to_tsquery('simple', :ts_query)
            )                  AS fts_rank,
            metadata
        FROM doc_embeddings
        WHERE is_deleted = FALSE
          AND (farm_id = CAST(:farm_id AS uuid) OR farm_id IS NULL)
          {type_clause}
          AND to_tsvector('simple', chunk_text) @@ websearch_to_tsquery('simple', :ts_query)
        ORDER BY fts_rank DESC
        LIMIT :limit
    """)
    fts_rows = (await session.execute(fts_sql, params)).mappings().all()

    # 5. RRF Fusion
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

    # 6. Filter and Format
    results = []
    for item in sorted_chunks:
        row = item["row"]
        sim = row.get("similarity_score") or 0.0
        if sim < settings.RAG_SIMILARITY_THRESHOLD:
            continue
        results.append(RetrievalResult(
            chunk_id=row["chunk_id"],
            chunk_text=row["chunk_text"],
            source_file=row["source_file"],
            source_type=row["source_type"],
            chunk_index=row["chunk_index"],
            similarity_score=round(float(sim), 4),
            fts_rank=float(row.get("fts_rank") or 0),
            rrf_score=round(item["rrf"], 6),
            metadata=dict(row["metadata"] or {}),
        ))

    log.info("retrieval.done", query=query[:80], returned=len(results))
    return results
