from __future__ import annotations
from pathlib import Path
from uuid import UUID
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from app.embedding.embedder import Embedder
from app.ingest.loaders import load_file
from app.rag.chunker import Chunker
from app.core.config import settings
from app.debug.tracer import get_logger

log = get_logger(__name__)
embedder = Embedder()
chunker = Chunker()

async def ingest_document(
    session: AsyncSession,
    file_path: Path,
    source_type: str,
    farm_id: UUID | None = None,
) -> dict:
    filename = file_path.name
    log.info("ingest.start", file=filename, source_type=source_type)

    # 1. Soft-delete old chunks
    await session.execute(
        sa.text(
            "UPDATE doc_embeddings SET is_deleted = TRUE "
            "WHERE source_file = :filename AND (farm_id = :farm_id OR farm_id IS NULL)"
        ),
        {"filename": filename, "farm_id": farm_id},
    )

    # 2. Load and Chunk
    raw_text = load_file(file_path)
    chunks = chunker.chunk(raw_text, source_file=filename)

    # 3. Embed and Store
    texts = [c["chunk_text"] for c in chunks]
    embeddings = await embedder.embed_many(texts)

    import json
    if embeddings:
        log.info("ingest.embedding_info", dim=len(embeddings[0]), expected=settings.EMBEDDING_DIM)
    
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        # Chuyển list sang chuỗi format của pgvector: '[0.1, 0.2, ...]'
        embed_str = "[" + ",".join(map(str, embedding)) + "]"
        
        rows.append({
            "farm_id": str(farm_id) if farm_id else None,
            "source_file": filename,
            "source_type": source_type,
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"],
            "chunk_tokens": chunk["chunk_tokens"],
            "embedding": embed_str,
            "metadata": json.dumps({"source_file": filename, "source_type": source_type}, ensure_ascii=False),
        })

    await session.execute(
        sa.text("""
            INSERT INTO doc_embeddings
                (farm_id, source_file, source_type, chunk_index, chunk_text,
                 chunk_tokens, embedding, metadata)
            VALUES
                (:farm_id, :source_file, :source_type, :chunk_index, :chunk_text,
                 :chunk_tokens, CAST(:embedding AS public.vector), CAST(:metadata AS jsonb))
        """),
        rows,
    )
    await session.commit()

    summary = {
        "file": filename,
        "source_type": source_type,
        "chunks_ingested": len(chunks),
        "total_tokens": sum(c["chunk_tokens"] for c in chunks),
    }
    log.info("ingest.done", **summary)
    return summary
