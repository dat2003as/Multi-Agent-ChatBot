# Multi-Agent Chatbot

> Multi-agent AI chatbot for high-tech shrimp farming operations — combining RAG document retrieval with Text-to-SQL structured data queries.

Built with **FastAPI** + **Azure OpenAI** + **PostgreSQL (pgvector)** + **Vanna** + **Redis**

---

## Architecture

![System Architecture](app/img/FullDiagram.png)

### Request Flow

```
User Request → UUID Detection → Farm Context Resolution → Orchestrator (LLM Agentic Loop)
                                                              │
                        ┌─────────────┬──────────────┬────────┴────────┐
                        ▼             ▼              ▼                 ▼
                search_documents  query_data   answer_general   search_realtime_info
                  (RAG Pipeline)  (Vanna SQL)  (LLM Knowledge)  (Prices / Weather)
                        │             │              │                 │
                        └─────────────┴──────────────┴────────┬────────┘
                                                              ▼
                                                     Final Answer (Vietnamese)
```

### RBAC — Role-Based Access Control

![RBAC Detail](app/img/rbac-detail.png)

Chỉ **`query_data`** (Text-to-SQL) bị kiểm tra RBAC. Ba tool còn lại mở cho tất cả role.

| Role | A1 Farm Ops | A2 Inventory | A3 Analytics | A4 IoT |
|------|:-----------:|:------------:|:------------:|:------:|
| **farmer** | ✅ | ❌ | ❌ | ✅ |
| **employee** | ✅ | ✅ | ✅ | ✅ |
| **manager** | ✅ | ✅ | ✅ | ✅ |
| **admin** | ✅ | ✅ | ✅ | ✅ |

---

## Features

### 4 AI Tools

| Tool | Description | Data Source |
|------|-------------|------------|
| `search_documents` | RAG hybrid search — vector cosine + BM25 full-text, merged via RRF (α=0.7) | PostgreSQL + pgvector |
| `query_data` | Text-to-SQL via Vanna — auto-detect domain (A1–A4), generate & execute SQL | PostgreSQL BE DB |
| `answer_general` | Aquaculture expert knowledge — diseases, water quality, feeding, pond management | Azure OpenAI LLM |
| `search_realtime_info` | Real-time shrimp prices (tepbac.com) and weather forecast (wttr.in) | Web scraping / API |

### 4 Data Domains (Vanna Text-to-SQL)

| Domain | Schema | Coverage |
|--------|--------|----------|
| **A1** — Farm Operations | `vanna_a1` | Ponds, diseases, sensors, harvesting, feed |
| **A2** — Inventory | `vanna_a2` | Stock, receipts, materials, suppliers, purchase orders |
| **A3** — Analytics | `vanna_a3` | KPIs, costs, revenue, performance metrics |
| **A4** — IoT & Scales | `vanna_a4` | Devices, aerators, pumps, cameras, sensors, cabinets |

### Security

- **UUID Detection** — block data enumeration attempts
- **Farm Isolation** — `farm_id` filter enforced on all queries
- **SQL Guardrail** — SELECT-only, injection pattern blocking, farm_id verification
- **RLS Context** — `SET app.current_farm_id` on every connection
- **Cross-Farm Blocking** — LLM extracts farm mentions, verifies authorization
- **Anti-Hallucination** — hide UUIDs, table names, SQL from responses
- **Soft Delete** — `IsDeleted = false` check enforced

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Runtime** | Python 3.11, FastAPI, Uvicorn |
| **LLM Chat** | Azure OpenAI — `gpt-4.1-mini` |
| **Embeddings** | Azure OpenAI — `text-embedding-3-small` (1536 dims) |
| **Text-to-SQL** | Vanna 2.0 (monkey-patched HuggingFace → Azure embeddings) |
| **Vector DB** | PostgreSQL + pgvector (IVFFlat index, cosine similarity) |
| **Business DB** | PostgreSQL (async + sync pools, RLS) |
| **Session Memory** | Redis Cluster — 3-turn window, 30min TTL |
| **Observability** | structlog, in-memory request tracker, debug dashboards |

---

## Project Structure

```
mebieco-ai-chatbot/
├── app/
│   ├── main.py                    # FastAPI entry point, CORS, debug routes
│   ├── dependencies.py            # DB engine, Redis client, DI providers
│   ├── api/
│   │   ├── chat.py                # POST /api/v1/chat
│   │   ├── ingest.py              # POST /api/v1/ingest (file upload)
│   │   ├── farm_context.py        # Farm name resolution + authorization
│   │   └── shrimp_prices.py       # Shrimp price API for MB app (tepbac.com scraper)
│   ├── core/
│   │   ├── config.py              # Pydantic Settings (all env vars)
│   │   ├── llm_clients.py         # Azure OpenAI async client factory
│   │   ├── db_pool.py             # psycopg2 connection pool (sync)
│   │   └── permissions.py         # RBAC: role → allowed domains
│   ├── orchestrator/
│   │   ├── orchestrator.py        # LLM agentic loop (max 5 iterations)
│   │   ├── tool_registry.py       # 4 tool definitions (OpenAI function schema)
│   │   └── tool_executor.py       # Tool dispatch & execution
│   ├── rag/
│   │   ├── retriever.py           # Hybrid search: vector + FTS + RRF fusion
│   │   └── chunker.py             # Token-based splitting (800 tokens/chunk)
│   ├── vanna/
│   │   ├── agent.py               # VannaAgent — singleton per domain
│   │   ├── intent_detector.py     # LLM domain classification (A1–A4)
│   │   ├── domain_router.py       # farm_id → domain mapping
│   │   ├── guardrail.py           # SQL validation (SELECT-only, injection block)
│   │   ├── executor.py            # SQL execution (readonly, RLS, LIMIT 500)
│   │   └── error_logger.py        # CSV error logging
│   ├── embedding/
│   │   └── embedder.py            # Azure OpenAI embedding with retry/batch
│   ├── memory/
│   │   └── session_memory.py      # Redis-backed conversation history
│   ├── ingest/
│   │   ├── pipeline.py            # Load → Chunk → Embed → Store
│   │   └── loaders.py             # PDF, DOCX, MD, TXT, XLSX
│   ├── prompts/
│   │   ├── system_prompt.py       # Classification rules + RBAC + anti-hallucination
│   │   └── rag_prompt.py          # RAG synthesis instructions
│   ├── debug/
│   │   ├── request_tracker.py     # In-memory ring buffer (20 logs)
│   │   ├── tracer.py              # structlog configuration
│   │   ├── dashboard.html         # Debug dashboard UI
│   │   └── 3d_flow.html           # 3D flow visualization
│   └── img/                       # Screenshots & diagrams
├── scripts/
│   ├── init_db.py                 # Create doc_embeddings table (pgvector)
│   └── ingest_docs.py             # Batch ingest documents
├── tests/
├── docs/
│   ├── architecture.drawio
│   └── rbac-detail.drawio
├── requirements.txt
└── .env
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- Redis (local or Azure Redis Cluster)
- Azure OpenAI resource (chat + embedding deployments)

### Installation

```bash
# Clone repository
git clone https://github.com/dat2003as/multi-agent-chatbot.git
cd multi-agent-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# ─── Azure OpenAI — Chat ───────────────────────────────
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini

# ─── Azure OpenAI — Embeddings ─────────────────────────
AZURE_OPENAI_EMBED_ENDPOINT=https://your-embed-resource.openai.azure.com/
AZURE_OPENAI_EMBED_API_KEY=your-embed-api-key
AZURE_OPENAI_EMBED_API_VERSION=2024-08-01-preview
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small

# ─── PostgreSQL + pgvector (RAG vector store) ──────────
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

# ─── PostgreSQL — BE Database (business data) ──────────
BE_DB_URL=postgresql://user:pass@host:5432/dbname
PG_HOST=host
PG_DB=dbname
PG_USER=user
PG_PASSWORD=pass
PG_PORT=5432

# ─── Redis ──────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=1800

# ─── RAG Tuning ────────────────────────────────────────
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.35
RAG_HYBRID_ALPHA=0.7

# ─── Debug ──────────────────────────────────────────────
LOG_LEVEL=DEBUG
TRACE_TOOL_CALLS=true
TRACE_RETRIEVAL=true
```

### Database Setup

```bash
# Create pgvector table for RAG
python scripts/init_db.py

# Ingest documents (SOPs, guides, FAQs)
python scripts/ingest_docs.py
```

### Run

```bash
# Development (with hot reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Vanna Training

Vanna sử dụng 3 loại file để training cho mỗi domain:

| File Type | Purpose | Example |
|-----------|---------|---------|
| `.sql` | DDL schema + sample queries | `CREATE TABLE`, `SELECT ... JOIN` |
| `.csv` | Question-SQL pairs | `"Liệt kê ao nuôi","SELECT * FROM Tbl_Pond"` |
| `.md` | Documentation context | Business rules, field descriptions |

```bash
# A1 — Farm Operations
python Vanna_AO/trainVanaKhoThu2\ 1.py

# A2 — Inventory Warehouse
python trainVanaKhoThu2.py

# A4 — IoT & Scales
python A4_IoT_Device_Vanna/trainVanaIoTThu4.py
```

Training data được lưu vào PostgreSQL schemas riêng biệt (`vanna_a1` ~ `vanna_a4`), mỗi domain có vector context riêng.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/chat` | Main chat endpoint |
| `POST` | `/api/v1/ingest` | Document ingestion (file upload) |
| `GET` | `/shrimp-prices` | Shrimp price data for MB app (tepbac.com) |
| `GET` | `/debug-dashboard` | Debug dashboard UI |
| `GET` | `/3d-flow` | 3D flow visualization |
| `GET` | `/api/v1/debug/requests` | Debug request logs (JSON) |
| `DELETE` | `/api/v1/debug/requests` | Clear debug logs |
| `GET` | `/` | Health check |

### Chat Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-001",
    "farm_id": "your-farm-uuid",
    "role": "farmer",
    "message": "Hiện tại trong kho còn những vật tư gì vậy?"
  }'
```

### Response

```json
{
  "session_id": "test-session-001",
  "answer": "Hiện tại trong kho của Farm 5 Kiên Giang có các vật tư sau: ..."
}
```

---

## Debug & Observability

### Debug Dashboard

Theo dõi real-time các request, tool calls, token usage và response time.

**Document Search & General Knowledge:**

![Debug Dashboard — Documents](app/img/debug_dashboard_Documents.png)

**Shrimp Prices & Weather:**

![Debug Dashboard — Prices & Weather](app/img/debug_dashboard_GiaTom_ThoiTiet.png)

**Out-of-Scope & Query Data:**

![Debug Dashboard — Query Data](app/img/debug_dashboard_OOScope_QueryData.png)

**Weather & Document Search (SOP/Guides):**

![Debug Dashboard — Weather & Guides](app/img/debug_dashboard_ThoiTiet_HuongDan.png)

### 3D Flow Visualization

Trực quan hóa luồng xử lý request dưới dạng 3D graph.

![3D Flow View](app/img/3dlow.png)

---

## License

Internal project — technical documentation.
