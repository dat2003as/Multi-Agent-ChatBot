import tempfile
from pathlib import Path
from uuid import UUID
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile
from app.dependencies import get_db, verify_api_key
from app.ingest.pipeline import ingest_document
from app.debug.tracer import get_logger

from pydantic import BaseModel, Field

log = get_logger(__name__)
router = APIRouter(
    prefix="/ingest", 
    tags=["ingest"],
    dependencies=[Depends(verify_api_key)]
)

ALLOWED_TYPES = {".pdf", ".docx", ".md", ".txt", ".xlsx", ".xls"}

class IngestData(BaseModel):
    filename: str
    chunks_count: int = Field(0, description="Number of chunks created")
    source_type: str

class IngestResponse(BaseModel):
    status: str = Field("success", example="success")
    message: str = Field("Document ingested successfully", example="Document ingested successfully")
    data: Optional[IngestData] = None

@router.post("", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    source_type: str = Form(..., description="sop | faq | guide | manual"),
    farm_id: UUID | None = Form(None),
    db=Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_TYPES:
        return IngestResponse(
            status="error",
            message=f"Unsupported file type: {suffix}",
            data=None
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    log.info("api.ingest", filename=file.filename, source_type=source_type)
    summary = await ingest_document(
        session=db,
        file_path=tmp_path,
        source_type=source_type,
        farm_id=farm_id,
    )
    tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        status="success",
        message="Document ingested successfully",
        data=IngestData(
            filename=file.filename or "unknown",
            chunks_count=summary.get("chunks_ingested", 0),
            source_type=source_type
        )
    )
