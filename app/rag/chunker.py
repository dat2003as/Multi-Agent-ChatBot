from __future__ import annotations
import re
import tiktoken
from app.debug.tracer import get_logger

log = get_logger(__name__)

CHUNK_TARGET_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 150
CHUNK_MAX_TOKENS = 1024

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
        paragraphs = self._split_paragraphs(text)
        chunks: list[dict] = []
        current_parts: list[str] = []
        current_tokens = 0
        overlap_buffer = ""

        for para in paragraphs:
            para_tokens = self._count(para)
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

        log.info("chunker.done", source=source_file, total_chunks=len(chunks))
        return chunks

    def _flush(self, chunks: list, parts: list[str], overlap: str) -> None:
        text = (overlap + "\n\n" + "\n\n".join(parts)).strip()
        chunks.append({
            "chunk_index": len(chunks),
            "chunk_text": text,
            "chunk_tokens": self._count(text),
        })

    def _make_overlap(self, parts: list[str]) -> str:
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
