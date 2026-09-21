# 📚 OnushilonAI - Exam Preparation RAG Application

A Retrieval-Augmented Generation (RAG) system that turns a pile of course PDFs into an AI study partner — one that actually knows what's in *your* lecture notes and tells you exactly where it found the answer.

Upload your slides, textbook chapters, or lecture PDFs. Ask a question in plain English. Get back an answer grounded in your own material, with the filename and page number cited — no hallucinated facts, no generic textbook answers pulled from nowhere.

> Built as a portfolio project to explore practical, production-style RAG architecture — not just a notebook demo, but a real API with persistence, deployment, and cost constraints taken seriously.

---

## Why this exists

Most "chat with your PDF" demos are Jupyter notebooks that die the moment the kernel restarts. This project started as one of those (a Kaggle notebook prototype) and was deliberately rebuilt as a real service: a FastAPI backend with a persistent vector index, a lightweight frontend, and a deployment path that costs nothing to run — because the whole point was to make it usable by actual students, not just impressive in a demo.

## How it works

1. **Ingest** — A course PDF is parsed page-by-page with PyMuPDF, so every extracted chunk of text remembers exactly which page it came from.
2. **Chunk** — Text is split into overlapping word-windows, small enough for precise retrieval, large enough to preserve context.
3. **Embed** — Each chunk is converted into a dense vector using a BGE embedding model, run through ONNX Runtime rather than a full PyTorch stack — enough of a memory difference to matter when you're deploying on a free-tier server.
4. **Index** — Vectors are stored in a FAISS index, saved to disk per course, so nothing needs to be re-processed after a restart.
5. **Retrieve & Answer** — A student's question is embedded and matched against the index; the most relevant chunks are handed to an LLM with strict instructions to answer *only* from what was retrieved, citing the source page.

```
PDF Upload → Extract (PyMuPDF) → Chunk → Embed (fastembed/ONNX) → FAISS Index (persisted)
                                                                          │
Student Question → Embed → Similarity Search ──────────────────────────┘
                                    │
                                    ▼
                        Retrieved Chunks + Question
                                    │
                                    ▼
                          LLM (Groq API) → Grounded Answer + Page Citations
```

## Features

- **Multi-course support** — organize materials by subject, each with its own isolated index
- **Grounded answers only** — the model is instructed to say "not found in the materials" rather than guess
- **Page-level citations** — every answer points back to the exact document and page
- **Persistent by design** — indexes survive server restarts; no reprocessing PDFs on every boot
- **Zero-cost to run** — no paid APIs, no GPU requirement, deployable on free-tier hosting
- **Swappable LLM backend** — use a free hosted API (Groq) or run fully offline with a local model (Ollama)

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Fast to build, automatic docs, async-friendly |
| PDF parsing | PyMuPDF | Reliable text + page-number extraction |
| Embeddings | fastembed (BGE, ONNX Runtime) | Same quality as sentence-transformers, a fraction of the memory — no PyTorch required |
| Vector search | FAISS | Fast, battle-tested, no external database to manage |
| LLM | Groq API (free tier) / Ollama (local, optional) | Fast inference with zero API cost |
| Frontend | Vanilla HTML/JS | No build step, easy to swap out |
| Deployment | Docker → Render | Free hosting, reproducible builds |

## Getting started

```bash
git clone <your-repo-url>
cd exam-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add a free API key from https://console.groq.com into .env

uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` for the web UI, or `http://localhost:8000/docs` for interactive API documentation.

## API overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/courses` | Create a new course |
| `GET` | `/courses` | List all courses |
| `POST` | `/courses/{id}/upload` | Upload and index a PDF into a course |
| `POST` | `/courses/{id}/ask` | Ask a question, get a grounded answer with sources |

## Deployment

The project ships with a `Dockerfile` and deploys cleanly to any container-friendly free host (Render is the tested path). Set `GROQ_API_KEY` and `GROQ_MODEL` as environment variables on your host, and make sure the app binds to the port your platform provides (see `Dockerfile` for how `$PORT` is handled). Note that free-tier instances typically cap memory around 512MB and spin down after inactivity — both were design constraints while building this, not afterthoughts.

## Roadmap

- [ ] Cross-encoder re-ranking on top of vector retrieval, for a meaningful accuracy jump
- [ ] OCR support for scanned PDFs and slide decks with no text layer
- [ ] A small evaluation set (question → known correct page) to tune retrieval with numbers, not vibes
- [ ] Basic auth / per-student accounts for real classroom use

## Lessons learned

Moving this from a Kaggle notebook to a deployed service surfaced problems a notebook never forces you to solve: state that needs to survive a restart, a GPU that won't be there in production, and a memory budget that a free hosting tier enforces whether you planned for it or not. Working through those constraints — like swapping a PyTorch-based embedding model for an ONNX-based one to fit inside a 512MB memory limit — ended up being as much a part of building this as the RAG pipeline itself.

MIT — use it, fork it, learn from it.
