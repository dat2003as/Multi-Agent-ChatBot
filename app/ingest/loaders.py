"""
Bộ tải file: PDF, DOCX, Markdown, Excel → văn bản thuần
"""
from __future__ import annotations
from pathlib import Path
import PyPDF2
import docx as python_docx
import pandas as pd
from app.debug.tracer import get_logger

log = get_logger(__name__)

def load_pdf(path: Path) -> str:
    reader = PyPDF2.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
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

def load_excel(path: Path) -> str:
    # Đọc tất cả các sheet và chuyển thành chuỗi văn bản
    excel_data = pd.read_excel(path, sheet_name=None)
    content_parts = []
    for sheet_name, df in excel_data.items():
        content_parts.append(f"Sheet: {sheet_name}\n{df.to_string(index=False)}")
    full_text = "\n\n".join(content_parts)
    log.info("loader.excel.done", path=str(path), sheets=len(excel_data))
    return full_text

def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".md": load_markdown,
        ".txt": load_markdown,
        ".xlsx": load_excel,
        ".xls": load_excel
    }
    if suffix not in loaders:
        raise ValueError(f"Định dạng file không hỗ trợ: {suffix}")
    return loaders[suffix](path)
