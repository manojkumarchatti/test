import logging
import time
from typing import List, Optional

from cachetools import TTLCache
from fastapi import HTTPException
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

from app.core.config import get_settings
from app.models.schemas import AnswerResponse, QuestionRequest, SourceAttribution

logger = logging.getLogger(__name__)

_prompt = PromptTemplate(
    template=(
        "You are a company knowledge base assistant. Use the provided context snippets to "
        "answer concisely with citations only from the snippets. If the answer is unknown, "
        "say you do not know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    input_variables=["context", "question"],
)


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, docs: List[Document]) -> List[float]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:  # pragma: no cover - dependency optional
                logger.warning("Reranker unavailable: %s", exc)
                return [0.0 for _ in docs]
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)


class RetrievalService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache: TTLCache[str, AnswerResponse] = TTLCache(maxsize=128, ttl=self.settings.cache_ttl_seconds)
        self.reranker = Reranker(self.settings.reranker_model)

    @property
    def _embeddings(self):
        from langchain.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _load_store(self) -> FAISS:
        try:
            return FAISS.load_local(
                str(self.settings.vector_store_path),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.error("Vector store missing: %s", exc)
            raise HTTPException(status_code=404, detail="No knowledge base has been ingested yet")

    def _retrieve(self, question: str, top_k: int, mmr_k: int) -> List[Document]:
        store = self._load_store()
        try:
            results = store.max_marginal_relevance_search(question, k=top_k, fetch_k=max(top_k * 2, mmr_k))
            logger.info("Retrieved %d documents using MMR", len(results))
            return results
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise HTTPException(status_code=500, detail="Retrieval failed")

    def _rerank(self, query: str, docs: List[Document], use_reranker: bool) -> List[Document]:
        if not use_reranker or not docs:
            return docs
        scores = self.reranker.score(query, docs)
        if not any(scores):
            return docs
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked]

    def _build_context(self, docs: List[Document]) -> str:
        pieces = []
        for idx, doc in enumerate(docs, start=1):
            snippet = doc.page_content.strip().replace("\n", " ")
            pieces.append(f"[{idx}] {snippet}")
        return "\n".join(pieces)

    def _llm(self):  # pragma: no cover - runtime dependency
        from langchain.llms import OpenAI

        if not self.settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not configured; returning citation-only summary")
            return None
        return OpenAI(temperature=0.1)

    def answer(self, request: QuestionRequest) -> AnswerResponse:
        cache_key = f"{request.question}-{request.top_k}-{request.use_reranker}"
        if cache_key in self.cache:
            logger.info("Cache hit for question: %s", request.question)
            cached = self.cache[cache_key]
            return AnswerResponse(**cached.dict(), from_cache=True)

        docs = self._retrieve(request.question, request.top_k, request.mmr_k)
        reranked = self._rerank(request.question, docs, request.use_reranker)
        context = self._build_context(reranked)

        llm = self._llm()
        if llm:
            chain = LLMChain(prompt=_prompt, llm=llm)
            answer_text = chain.run(context=context, question=request.question)
        else:
            snippets = " ".join(doc.page_content[:180] for doc in reranked[:request.top_k])
            answer_text = f"Based on the knowledge base, here is a summary: {snippets}".strip()

        sources: List[SourceAttribution] = []
        for idx, doc in enumerate(reranked[: request.top_k]):
            sources.append(
                SourceAttribution(
                    snippet=doc.page_content[:400],
                    metadata=doc.metadata,
                    score=1.0 / (idx + 1),
                )
            )

        response = AnswerResponse(answer=answer_text, sources=sources, from_cache=False)
        self.cache[cache_key] = response
        return response


retrieval_service = RetrievalService()
