# Company Knowledge Base RAG

Production-ready RAG scaffolding with FastAPI + LangChain backend and React front-end.

## Features
- PDF/HTML/Markdown ingestion → text cleaning → chunking.
- FAISS vector store with MMR retrieval and optional cross-encoder reranking.
- Citation-backed answers plus caching and simple in-memory rate limiting.
- Observability via structured logging middleware.
- Dockerfiles for backend/frontend and a docker-compose for local orchestration.

## Running locally
```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Set `OPENAI_API_KEY` to enable LLM-backed generation; otherwise citation-only summaries are returned.

## API
- `POST /api/ingest` with `multipart/form-data` file to index.
- `POST /api/query` with `{ "question": "..." }` returns `answer`, `sources`, and `from_cache` flag.
