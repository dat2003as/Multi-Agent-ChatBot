"""
app/vanna/agent.py
==================
VannaAgent — Tích hợp hệ sinh thái chuẩn Vanna (Legacy SDK)
Hỗ trợ Dynamic Domain Routing (A1, A2, A3 qua Postgres Schema).
Bảo mật tuyệt đối: Luôn inject `farm_id` vào mọi lệnh sinh SQL.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# -----------------------------------------------------------------------------
# BƯỚC 0: Import config TRƯỚC để lấy thông số embed
# -----------------------------------------------------------------------------
from app.core.config import settings

# -----------------------------------------------------------------------------
# BƯỚC 1: Tạo Azure LangChain Embeddings Wrapper
# PHẢI làm TRƯỚC KHI import Vanna
# Dùng ĐÚNG endpoint/key/version của embed resource (KHÁC với chat resource)
# -----------------------------------------------------------------------------
from langchain_openai import AzureOpenAIEmbeddings

_EMBED_MODEL    = settings.AZURE_OPENAI_EMBED_DEPLOYMENT                             
_EMBED_ENDPOINT = settings.AZURE_OPENAI_EMBED_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT
_EMBED_KEY      = settings.AZURE_OPENAI_EMBED_API_KEY  or settings.AZURE_OPENAI_API_KEY
_EMBED_VERSION  = settings.AZURE_OPENAI_EMBED_API_VERSION                            

azure_lc_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=_EMBED_ENDPOINT, 
    api_key=_EMBED_KEY,              
    api_version=_EMBED_VERSION,     
    azure_deployment=_EMBED_MODEL,   
)

# -----------------------------------------------------------------------------
# MONKEY PATCH 1: Chặn HuggingFace tại gốc
# Vanna legacy gọi HuggingFaceEmbeddings() khi khởi tạo PGVector.
# Ta thay toàn bộ class đó bằng Azure wrapper TRƯỚC KHI Vanna import.
# -----------------------------------------------------------------------------
try:
    import langchain_community.embeddings as _lce
    _lce.HuggingFaceEmbeddings = lambda *args, **kwargs: azure_lc_embeddings
except ImportError:
    pass

try:
    import langchain_huggingface as _lhf
    _lhf.HuggingFaceEmbeddings = lambda *args, **kwargs: azure_lc_embeddings
except ImportError:
    pass

# -----------------------------------------------------------------------------
# MONKEY PATCH 2: Bỏ qua CREATE EXTENSION (thiếu quyền admin Azure PG)
# MONKEY PATCH 3: Inject Azure embeddings vào LangChain PGVector __init__
# -----------------------------------------------------------------------------
import langchain_postgres.vectorstores as _lpv

_lpv.PGVector.create_vector_extension = lambda *args, **kwargs: None

_original_pgvector_init = _lpv.PGVector.__init__

def _patched_pgvector_init(self, *args, **kwargs):
    kwargs["embeddings"] = azure_lc_embeddings  # Ghi đè bất kể giá trị cũ
    _original_pgvector_init(self, *args, **kwargs)

_lpv.PGVector.__init__ = _patched_pgvector_init

# -----------------------------------------------------------------------------
# SAU KHI ĐÃ PATCH XONG → mới import Vanna
# -----------------------------------------------------------------------------
from openai import AzureOpenAI
from vanna.legacy.openai.openai_chat import OpenAI_Chat
from vanna.legacy.openai.openai_embeddings import OpenAI_Embeddings
from vanna.legacy.pgvector.pgvector import PG_VectorStore

from app.vanna.guardrail import GuardrailError, validate_sql
from app.vanna.executor import execute_sql

logger = logging.getLogger(__name__)

_STOCK_KEYWORDS = re.compile(r'tồn kho|ton kho|trong kho|còn kho', re.IGNORECASE)
_HAS_QTY_FILTER = re.compile(r'"Quantity"\s*>\s*0', re.IGNORECASE)
_ORDER_LIMIT_PAT = re.compile(r'(ORDER\s+BY|LIMIT|;)\s', re.IGNORECASE)

def _ensure_stock_filter(sql: str, question: str) -> str:
    if not _STOCK_KEYWORDS.search(question):
        return sql
    if 'Tbl_WarehouseItem' not in sql:
        return sql
    if _HAS_QTY_FILTER.search(sql):
        return sql
    match = _ORDER_LIMIT_PAT.search(sql)
    if match:
        inject_pos = match.start()
        sql = sql[:inject_pos] + '  AND wi."Quantity" > 0\n' + sql[inject_pos:]
    else:
        sql = sql.rstrip().rstrip(';') + '\n  AND wi."Quantity" > 0;'
    logger.info("[StockFilter] Injected Quantity > 0 filter")
    return sql


# ─────────────────────────────────────────────────────────────────────────────
# TẦNG 1 — CRITICAL RULES
# ─────────────────────────────────────────────────────────────────────────────
_CRITICAL_RULES = """## ⚠️ CRITICAL RULES — ABSOLUTELY MANDATORY (no exceptions)

### SECURITY
C1. Write SELECT queries only. NEVER use INSERT / UPDATE / DELETE / DROP / TRUNCATE / ALTER.
C2. ALWAYS inject w."ZoneId" = '{farm_id}' when query has JOIN with Tbl_Warehouse. No exceptions.
C3. NEVER cross-farm: use only the assigned farm_id, never hardcode other UUIDs.
C4. Tbl_CentralWarehouse is system-wide data — DO NOT filter by ZoneId.

### SOFT DELETE
C5. ALWAYS add "IsDeleted" = false for ALL tables in FROM and JOIN clauses — no exceptions.

### BUSINESS LOGIC
C6. When calculating quantity/amount for imports/exports: only count Status = 'Approved'.
C7. Actual stock = Tbl_WarehouseItem.Quantity (real-time). DO NOT sum import/export receipts.
C8. Stock value = Quantity * AveragePrice (weighted average costing).
C9. Low stock alert: Quantity < AlertQty AND AlertQty > 0.
C10. For "manual creation" queries: filter AutoGenerated = false.
C11. When user asks for average/mean price: return INDIVIDUAL items with their prices (do NOT use AVG aggregate). Let the chatbot calculate the average.
C12. When user asks "bao nhiêu vật tư/loại" (how many items/types): use COUNT(DISTINCT), NOT SUM(Quantity). SUM is only for total stock quantity.

### MATERIAL CATALOG (Tbl_Material)
C13. "Code" column = Mã vật tư, starts with "N-VT" prefix (e.g. N-VT0127). "Name" column = Tên vật tư / product name (e.g. "9832 - SEA-TOM"). NEVER confuse Code with Name.
C14. When user asks to filter by material name: use m."Name" ILIKE '%keyword%', NOT m."Code".
C15. When user asks for material details/description: JOIN Tbl_MaterialGroup, Tbl_MaterialType, Tbl_Unit and return full columns:
     SELECT m."Code", m."Name", mg."Name" AS "Nhóm vật tư", mt."Name" AS "Loại vật tư", m."Description" AS "Mô tả", m."Manufacturer" AS "Nhãn hàng", u."Name" AS "Đơn vị tính", m."Dosage", m."UsageInstruction"
     FROM public."Tbl_Material" m JOIN public."Tbl_MaterialGroup" mg ON mg."Id" = m."MaterialGroupId" JOIN public."Tbl_MaterialType" mt ON mt."Id" = m."MaterialTypeId" JOIN public."Tbl_Unit" u ON u."Id" = m."UnitId"

### SQL SYNTAX
C16. All table and column names use double-quotes with PascalCase.
C17. All ForeignKeys are UUIDs — DO NOT JOIN using integers.
C18. Return ONLY the SQL query as pure text. No markdown, no backticks, no explanations.
"""


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeVanna (Core Vanna Wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class RuntimeVanna(PG_VectorStore, OpenAI_Chat, OpenAI_Embeddings):
    """Class lõi gộp 3 chức năng: Vector DB, Chat LLM, Embeddings."""

    def __init__(self, config=None):
        # Khởi tạo Não (LLM + Embeddings) TRƯỚC
        OpenAI_Chat.__init__(self, client=config.get("chat_client"), config=config)
        OpenAI_Embeddings.__init__(self, client=config.get("embed_client"), config=config)
        # Khởi tạo Bộ Nhớ (VectorStore) SAU CÙNG
        PG_VectorStore.__init__(self, config=config)

    # ✅ Override generate_embedding — đảm bảo Vanna dùng Azure khi train + query
    def generate_embedding(self, data: str, **kwargs) -> list[float]:
        result = self.config["embed_client"].embeddings.create(
            model=self.config.get("engine", "text-embedding-3-small"),
            input=data,
        )
        return result.data[0].embedding  # Đúng cú pháp OpenAI SDK v1+


# ─────────────────────────────────────────────────────────────────────────────
# VannaAgent (Controller)
# ─────────────────────────────────────────────────────────────────────────────
class VannaAgent:
    def __init__(self, domain_key: str = "a2") -> None:
        """
        Khởi tạo Agent gắn liền với một Domain để trỏ đúng Schema DB.
        domain_key có thể là: 'a1', 'a2', 'a3'.
        """
        self.domain_key = domain_key.lower()
        self.schema_name = f"vanna_{self.domain_key}"
        logger.info("[VannaAgent] init domain=%s schema=%s", domain_key.upper(), self.schema_name)

        # 1. Setup Clients
        # ✅ embed_client: dùng ĐÚNG endpoint/key/version của embed resource
        embed_client = AzureOpenAI(
            azure_endpoint=_EMBED_ENDPOINT,  
            api_key=_EMBED_KEY,             
            api_version=_EMBED_VERSION,      
        )
        # ✅ chat_client: dùng endpoint/key/version của chat resource (GPT-4.1-mini)
        chat_client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

        # 2. Setup DB Connection (Isolation via Schema)
        import urllib.parse
        from sqlalchemy import create_engine, event

        encoded_pwd = urllib.parse.quote_plus(settings.PG_PASSWORD)
        db_url = (
            f"postgresql+psycopg2://{settings.PG_USER}:{encoded_pwd}"
            f"@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DB}?sslmode=require"
        )
        engine = create_engine(
            db_url,
            connect_args={"options": f"-csearch_path={self.schema_name},public"}
        )

        @event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute(f"SET search_path TO {self.schema_name}, public")
            cursor.close()

        # 3. Init Native Vanna
        self.vn = MebiecoRuntimeVanna(config={
            "connection_string": engine,
            "embed_client": embed_client,       # 
            "chat_client":  chat_client,        # 
            "model":        settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            "engine":       _EMBED_MODEL,       # 
        })

    # ─────────────────────────────────────────────────────────────────────────
    def query(self, question: str, farm_id: str, user_id: str | None = None, original_question: str | None = None) -> dict[str, Any]:
        """User đặt câu hỏi -> RAG -> LLM -> Guardrail -> Execute"""
        logger.info(
            "[VannaAgent] query domain=%s schema=%s",
            self.domain_key.upper(), self.schema_name
        )
        max_retries = 1
        acc_prompt_tokens = 0
        acc_completion_tokens = 0
        try:
            # BƯỚC 1: Lấy Vanna Context (Vector search) — chạy song song 3 calls
            with ThreadPoolExecutor(max_workers=3) as pool:
                f_sqls = pool.submit(self.vn.get_similar_question_sql, question)
                f_ddls = pool.submit(self.vn.get_related_ddl, question)
                f_docs = pool.submit(self.vn.get_related_documentation, question)

            try:
                q_sqls = f_sqls.result()
            except Exception as e:
                logger.warning("[VannaAgent] get_similar_question_sql failed: %s", e)
                q_sqls = []

            try:
                ddls = f_ddls.result()
            except Exception as e:
                logger.warning("[VannaAgent] get_related_ddl failed: %s", e)
                ddls = []

            try:
                docs = f_docs.result()
            except Exception as e:
                logger.warning("[VannaAgent] get_related_documentation failed: %s", e)
                docs = []

            # BƯỚC 2: Ráp Prompt Siêu Bảo Mật
            prompt = self._build_secure_prompt(question, q_sqls, ddls, docs, farm_id)

            for attempt in range(max_retries + 1):
                llm_res = None
                try:
                    # BƯỚC 3: Call LLM Chat trực tiếp
                    response = self.vn.config["chat_client"].chat.completions.create(
                        model=self.vn.config.get("model", settings.AZURE_OPENAI_CHAT_DEPLOYMENT),
                        messages=prompt,
                        temperature=0,
                    )
                    llm_res = response.choices[0].message.content

                    # Ghi Log chi phí Token
                    usage = response.usage
                    acc_prompt_tokens += usage.prompt_tokens
                    acc_completion_tokens += usage.completion_tokens
                    
                    print(f"[Vanna] Domain: {self.domain_key.upper()} | Attempt {attempt+1} | Total tokens: {usage.total_tokens}")
                    logger.info(
                        "[VannaAgent] tokens=%d attempt=%d",
                        usage.total_tokens, attempt + 1
                    )

                    # Bóc tách SQL
                    raw_sql = self.vn.extract_sql(llm_res)

                    # BƯỚC 4: Guardrail
                    safe_sql = validate_sql(raw_sql, farm_id=farm_id)

                    # BƯỚC 4.5: Auto-inject Quantity > 0 for stock queries
                    safe_sql = _ensure_stock_filter(safe_sql, original_question or question)

                    # BƯỚC 5: Thực thi
                    data, columns = execute_sql(safe_sql, farm_id=farm_id)

                    print(f"[Vanna] SQL executed: {safe_sql[:80]}...")
                    logger.info("[VannaAgent] executed rows=%d", len(data))
                    return {"sql": safe_sql, "data": data, "columns": columns, "error": None, "prompt_tokens": acc_prompt_tokens, "completion_tokens": acc_completion_tokens}

                except Exception as exc:
                    if isinstance(exc, GuardrailError):
                        logger.warning("[VannaAgent] guardrail blocked: %s", exc)
                        return {"sql": None, "data": [], "columns": [], "error": f"Guardrail: {exc}", "prompt_tokens": acc_prompt_tokens, "completion_tokens": acc_completion_tokens}

                    error_msg = str(exc)
                    if attempt < max_retries and llm_res is not None:
                        logger.warning("[VannaAgent] SQL execution error on attempt %d: %s. Retrying...", attempt + 1, error_msg)
                        prompt.append({"role": "assistant", "content": llm_res})
                        prompt.append({
                            "role": "user",
                            "content": f"The executed SQL query failed with this database error:\n{error_msg}\n\nPlease fix the SQL query based on the schema provided. Return ONLY the valid SQL query."
                        })
                    else:
                        logger.exception("[VannaAgent] unexpected error after retries: %s", exc)
                        return {"sql": None, "data": [], "columns": [], "error": error_msg, "prompt_tokens": acc_prompt_tokens, "completion_tokens": acc_completion_tokens}

        except Exception as exc:
            logger.exception("[VannaAgent] unexpected error during query setup: %s", exc)
            return {"sql": None, "data": [], "columns": [], "error": str(exc), "prompt_tokens": acc_prompt_tokens, "completion_tokens": acc_completion_tokens}

    # ─────────────────────────────────────────────────────────────────────────
    def _build_secure_prompt(
        self,
        question: str,
        q_sqls: list,
        ddls: list,
        docs: list,
        farm_id: str,
    ) -> list[dict]:
        """Ráp Prompt với CRITICAL RULES và đúng farm_id."""
        critical_block = _CRITICAL_RULES.replace("{farm_id}", farm_id)

        ddl_block = "\n\n".join(ddls)   if ddls   else "No context."
        doc_block = "\n\n".join(docs)   if docs   else ""
        sql_block = "\n\n".join(
            [f"Q: {q['question']}\nSQL: {q['sql']}" for q in q_sqls]
        ) if q_sqls else ""

        system_msg = (
            f"You are an AI Data Analyst for Mebieco. Active Farm_ID: '{farm_id}'.\n\n"
            f"{critical_block}\n\n"
            f"### DDL SCHEMA ###\n{ddl_block}\n\n"
            f"### RULE DOCUMENTATION ###\n{doc_block}\n\n"
        )
        user_msg = (
            f"### SIMILAR EXAMPLES ###\n{sql_block}\n\n"
            f"### ACTUAL QUESTION ###\n{question}\n\n"
            f"Generate PostgreSQL query. ALWAYS FILTER by ZoneId='{farm_id}'. Return SQL only (no explanation)."
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Factory Manager (Singleton per Domain)
# ─────────────────────────────────────────────────────────────────────────────
_agent_instances: dict[str, VannaAgent] = {}

def get_vanna_agent(domain_key: str = "a2") -> VannaAgent:
    """Singleton Factory: Tự động chia luồng A1, A2, A3 dựa trên domain_key."""
    k = domain_key.lower()
    if k not in _agent_instances:
        _agent_instances[k] = VannaAgent(domain_key=k)
    return _agent_instances[k]