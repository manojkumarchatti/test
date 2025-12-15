import hashlib
import io
import logging
import uuid
from pathlib import Path
from typing import List

import faiss  # type: ignore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pdfminer.high_level import extract_text as pdf_extract_text
from pydantic import BaseModel
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:12]


def _as_document(text: str, source: str) -> Document:
    return Document(page_content=text, metadata={"source": source})


class IngestionSummary(BaseModel):
    document_id: str
    chunks: List[Document]


class IngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _load_pdf(self, content: bytes, source: str) -> str:
        logger.info("Extracting text from PDF: %s", source)
        return pdf_extract_text(io.BytesIO(content))

    def _load_html(self, content: bytes, source: str) -> str:
        logger.info("Extracting text from HTML: %s", source)
        elements = partition_html(text=content.decode("utf-8", errors="ignore"))
        return "\n".join(element.text for element in elements if element.text)

    def _load_markdown(self, content: bytes, source: str) -> str:
        logger.info("Extracting text from Markdown: %s", source)
        elements = partition_text(text=content.decode("utf-8", errors="ignore"))
        return "\n".join(element.text for element in elements if element.text)

    def load_document(self, filename: str, content: bytes) -> IngestionSummary:
        suffix = Path(filename).suffix.lower()
        doc_id = f"doc-{_hash_content(content)}"
        target_path = self.settings.uploads_path / f"{doc_id}{suffix}"
        target_path.write_bytes(content)

        if suffix == ".pdf":
            text = self._load_pdf(content, filename)
        elif suffix in {".html", ".htm"}:
            text = self._load_html(content, filename)
        else:
            text = self._load_markdown(content, filename)

        chunker = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = chunker.split_documents([_as_document(text, str(target_path))])
        logger.info("Chunked %s into %d segments", filename, len(docs))
        return IngestionSummary(document_id=doc_id, chunks=docs)

    def persist_embeddings(self, docs: List[Document]) -> int:
        logger.info("Indexing %d chunks into FAISS store", len(docs))
        vector_store = self._load_or_create_store()
        vector_store.add_documents(docs)
        vector_store.save_local(str(self.settings.vector_store_path))
        return len(docs)

    def _load_or_create_store(self) -> FAISS:
        if self.settings.vector_store_path.exists():
            logger.info("Loading existing FAISS index from %s", self.settings.vector_store_path)
            return FAISS.load_local(
                str(self.settings.vector_store_path),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )
        logger.info("Creating new FAISS index")
        index = faiss.IndexFlatL2(self._embeddings.embed_query("test").shape[0])
        return FAISS(embedding_function=self._embeddings, index=index)

    @property
    def _embeddings(self):  # lazy import
        from langchain.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_file(filename: str, content: bytes) -> IngestionSummary:
    service = IngestionService()
    return service.load_document(filename, content)
