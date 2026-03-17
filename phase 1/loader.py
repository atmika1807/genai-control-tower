"""
Document loader
Resolves a source URI → raw text, regardless of origin.
Supports: local files, S3, HTTP/HTTPS, PDF, DOCX, plain text.
"""

import io
import httpx
import structlog
from pathlib import Path
from shared.models import SourceType

log = structlog.get_logger(__name__)


class DocumentLoader:

    async def load(self, source_type: SourceType, source_uri: str) -> str:
        """Return extracted plain text from any supported source."""
        log.info("loading_document", source_type=source_type, uri=source_uri)

        raw_bytes = await self._fetch_bytes(source_type, source_uri)
        return self._extract_text(source_type, source_uri, raw_bytes)

    # ── Fetching ──────────────────────────────────────────────────────────────

    async def _fetch_bytes(self, source_type: SourceType, uri: str) -> bytes:
        if source_type == SourceType.URL:
            return await self._fetch_http(uri)
        if source_type == SourceType.S3:
            return await self._fetch_s3(uri)
        # Local path (PDF / DOCX / TXT)
        return Path(uri).read_bytes()

    async def _fetch_http(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def _fetch_s3(self, s3_uri: str) -> bytes:
        # boto3 is synchronous — run in a thread pool to avoid blocking the event loop
        import asyncio
        import boto3

        def _sync_fetch(uri: str) -> bytes:
            # s3://bucket/key/path
            parts = uri.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            buf = io.BytesIO()
            s3.download_fileobj(bucket, key, buf)
            return buf.getvalue()

        return await asyncio.get_event_loop().run_in_executor(
            None, _sync_fetch, s3_uri
        )

    # ── Extraction ────────────────────────────────────────────────────────────

    def _extract_text(
        self, source_type: SourceType, uri: str, data: bytes
    ) -> str:
        # Infer from extension when source_type is URL or S3
        resolved = source_type
        if source_type in (SourceType.URL, SourceType.S3):
            suffix = Path(uri.split("?")[0]).suffix.lower()
            resolved = {
                ".pdf":  SourceType.PDF,
                ".docx": SourceType.DOCX,
                ".txt":  SourceType.TXT,
            }.get(suffix, SourceType.TXT)

        if resolved == SourceType.PDF:
            return self._extract_pdf(data)
        if resolved == SourceType.DOCX:
            return self._extract_docx(data)
        return data.decode("utf-8", errors="replace")

    def _extract_pdf(self, data: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[page {i + 1}]\n{text}")
        return "\n\n".join(pages)

    def _extract_docx(self, data: bytes) -> str:
        from docx import Document

        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
