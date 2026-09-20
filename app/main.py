"""
FastAPI service for the Exam Preparation AI.

Endpoints:
  POST /courses                 -> create a course
  GET  /courses                 -> list courses
  POST /courses/{id}/upload     -> upload + index a PDF into a course
  POST /courses/{id}/ask        -> ask a question, get a grounded answer + sources

Run locally:
  uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs, or open
frontend/index.html for the simple chat UI (point it at your server URL).
"""
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import SYSTEM_PROMPT, TOP_K
from app.course import create_course, list_courses, get_course
from app.llm import generate

app = FastAPI(title="Exam Preparation AI")

# Allow the frontend (served from anywhere, e.g. a static host) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateCourseRequest(BaseModel):
    name: str


class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K


def build_prompt(question: str, retrieved: List[dict]) -> str:
    context = "\n\n".join(
        f"[Chunk {i}] (Source: {r['filename']}, Page {r['page']})\n{r['text']}"
        for i, r in enumerate(retrieved, start=1)
    )
    return f"""Course materials:

{context}

Student question:
{question}

Provide a clear explanation suitable for exam preparation, using only the materials above."""


@app.post("/courses")
def api_create_course(req: CreateCourseRequest):
    course = create_course(req.name)
    return {"id": course.id, "name": course.name}


@app.get("/courses")
def api_list_courses():
    return list_courses()


@app.post("/courses/{course_id}/upload")
async def api_upload_pdf(course_id: str, file: UploadFile = File(...)):
    try:
        course = get_course(course_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Course '{course_id}' not found")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = course.add_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.post("/courses/{course_id}/ask")
def api_ask(course_id: str, req: AskRequest):
    try:
        course = get_course(course_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Course '{course_id}' not found")

    retrieved = course.retrieve(req.question, req.top_k)
    if not retrieved:
        return {"answer": "No material has been indexed for this course yet.", "sources": []}

    answer = generate(SYSTEM_PROMPT, build_prompt(req.question, retrieved))

    seen, sources = set(), []
    for r in retrieved:
        key = (r["filename"], r["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"filename": r["filename"], "page": r["page"], "score": round(r["score"], 3)})

    return {"answer": answer, "sources": sources}


# Serve the simple frontend at "/" (optional — remove if you host the
# frontend separately, e.g. on Vercel/Netlify pointing at this API).
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
