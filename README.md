# Exam Preparation AI — FastAPI service

Local, free-of-cost port of the Kaggle RAG notebook: upload course PDFs,
ask questions, get answers grounded in the material with filename + page
citations.

## What changed vs. the Kaggle version

| | Kaggle notebook | This app |
|---|---|---|
| LLM | 7B model on a free Kaggle GPU | Groq's free hosted API (or Ollama, fully local) — no GPU needed |
| State | Lives in notebook memory, lost on restart | FAISS index + chunk metadata saved to disk per course |
| Interface | Gradio, 72‑hour link | FastAPI + a small static frontend, deploy anywhere |

## 1. Run it locally

```bash
cd exam-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: paste in a free Groq API key (see below)

uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the chat UI, or **http://localhost:8000/docs**
for the interactive API docs.

### Getting a free Groq API key
1. Go to https://console.groq.com and sign up (no credit card).
2. Create an API key.
3. Put it in `.env` as `GROQ_API_KEY=...`. The free tier's rate limits are
   generous enough for a small class of students.

### Alternative: fully local LLM (zero external calls)
If you'd rather not call any external API at all:
1. Install [Ollama](https://ollama.com).
2. `ollama pull qwen2.5:7b-instruct` (or a smaller model if your machine is
   modest, e.g. `qwen2.5:1.5b-instruct`).
3. In `.env`, set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL` to match.

This runs entirely on whatever machine has Ollama installed — fine for local
testing or for hosting on your own server with decent RAM, but not realistic
on most free hosting tiers (no GPU, limited RAM), which is why Groq is the
default for step 2 below.

## 2. Use it

```bash
# create a course
curl -X POST localhost:8000/courses -H "Content-Type: application/json" -d '{"name": "Machine Learning"}'
# -> {"id": "machine-learning-a1b2c3", "name": "Machine Learning"}

# upload a PDF
curl -X POST localhost:8000/courses/machine-learning-a1b2c3/upload -F "file=@notes.pdf"

# ask a question
curl -X POST localhost:8000/courses/machine-learning-a1b2c3/ask \
  -H "Content-Type: application/json" -d '{"question": "What is overfitting?"}'
```

Or just use the web UI at `/` — create a course, upload a PDF, ask away.

## 3. Host it for free — Hugging Face Spaces (recommended)

Spaces gives you a free public URL, no credit card, using the included
`Dockerfile`.

1. Create a free account at https://huggingface.co.
2. **New Space** → pick a name → SDK: **Docker** → hardware: **CPU basic (free)**.
3. Push this project to the Space's git repo:
   ```bash
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git add .
   git commit -m "Initial commit"
   git push space main
   ```
4. In the Space's **Settings → Repository secrets**, add `GROQ_API_KEY`
   (same value as your local `.env`). Do **not** commit your `.env` file —
   it's already in `.gitignore`.
5. The Space builds the Dockerfile and gives you a URL like
   `https://<your-username>-<space-name>.hf.space` — share that with students.

Notes on the free CPU tier:
- Embeddings run on CPU there too — fine for a course's worth of PDFs
  (thousands of chunks), just slower than Kaggle's GPU for the initial
  indexing of a large PDF. Retrieval + LLM answering stays fast either way
  since the LLM call itself goes to Groq.
- Free Spaces sleep after a period of inactivity and wake on the next
  request (a few seconds' delay) — normal for a free tier.
- The `data/` directory persists across restarts as long as you don't
  delete/recreate the Space; for anything long-term-critical, Spaces also
  supports a small persistent disk add-on if you outgrow the default.

### Alternative free hosts
- **Render** (free web service tier) — same Dockerfile works, deploy via
  their dashboard connected to a GitHub repo. Free tier also sleeps after
  inactivity.
- **Railway** / **Fly.io** — both have small free allowances; same Docker
  image works unchanged.

## 4. Where to take this next

Same roadmap as the notebook's closing section:
1. Tune `CHUNK_SIZE_WORDS` / `TOP_K` in `.env` and compare answer quality.
2. Add a cross-encoder re-ranker (`BAAI/bge-reranker-base`) on top of the
   bi-encoder retrieval — usually the single biggest quality jump.
3. Add OCR (`pytesseract`) for scanned PDFs/slide decks with no text layer.
4. Build a small eval set (question → known correct page) to tune with
   numbers instead of vibes.
5. Add basic auth / per-student accounts once you're ready for real usage
   beyond a demo.
