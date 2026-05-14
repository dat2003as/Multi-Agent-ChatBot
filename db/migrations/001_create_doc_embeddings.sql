
-- Bảng embeddings
CREATE TABLE IF NOT EXISTS doc_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id         UUID,
    source_file     TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    chunk_tokens    INTEGER,
    embedding       public.vector(1536),
    metadata        JSONB DEFAULT '{}',
    is_deleted      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ✅ FIX: bỏ cosine_ops + bỏ hnsw
CREATE INDEX IF NOT EXISTS idx_doc_embed_vector
    ON doc_embeddings
    USING ivfflat (embedding)
    WITH (lists = 100);

-- Filter index
CREATE INDEX IF NOT EXISTS idx_doc_embed_farm
    ON doc_embeddings (farm_id);

CREATE INDEX IF NOT EXISTS idx_doc_embed_type
    ON doc_embeddings (source_type);

CREATE INDEX IF NOT EXISTS idx_doc_embed_deleted
    ON doc_embeddings (is_deleted)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_doc_embed_metadata
    ON doc_embeddings USING GIN (metadata);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_doc_embed_fts
    ON doc_embeddings
    USING GIN (to_tsvector('simple', chunk_text));

-- Trigger update time
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trg_doc_embeddings_updated_at ON doc_embeddings;

CREATE TRIGGER trg_doc_embeddings_updated_at
    BEFORE UPDATE ON doc_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 🔥 QUAN TRỌNG (bắt buộc)
ANALYZE doc_embeddings;