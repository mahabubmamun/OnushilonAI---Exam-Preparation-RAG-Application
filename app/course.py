"""
The Course object — mirrors the courses / documents / chunks tables from
the design doc. Unlike the notebook version, this one persists to disk
under DATA_DIR/<course_id>/ so the server can restart without losing
indexed courses.
"""
import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional

from app.config import DATA_DIR, TOP_K
from app.pdf_utils import extract_pages
from app.chunking import chunk_pages
from app.embeddings import embed_texts, embed_query
from app.vectorstore import VectorStore


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "course"


class Course:
    def __init__(self, course_id: str, name: str):
        self.id = course_id
        self.name = name
        self.store: Optional[VectorStore] = None
        self.chunks: List[Dict] = []   # parallel to vector ids: [{text, page, filename}]
        self.documents: List[str] = []

    # --- paths -----------------------------------------------------------
    @property
    def dir(self) -> Path:
        d = DATA_DIR / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def index_path(self) -> Path:
        return self.dir / "index.faiss"

    # --- indexing ----------------------------------------------------------
    def add_pdf(self, pdf_path: str) -> Dict:
        filename = Path(pdf_path).name
        pages = extract_pages(pdf_path)
        if not pages:
            return {"filename": filename, "status": "skipped (no extractable text — scanned PDF?)"}

        raw_chunks = chunk_pages(pages)
        vectors = embed_texts([c["text"] for c in raw_chunks])

        if self.store is None:
            self.store = VectorStore(dim=vectors.shape[1])

        self.store.add(vectors)
        for c in raw_chunks:
            self.chunks.append({"text": c["text"], "page": c["page"], "filename": filename})

        self.documents.append(filename)
        self.save()
        return {"filename": filename, "pages": len(pages), "chunks": len(raw_chunks)}

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        if self.store is None or self.store.size == 0:
            return []
        scores, ids = self.store.search(embed_query(question), top_k)
        results = []
        for score, idx in zip(scores, ids):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": float(score)})
        return results

    # --- persistence ---------------------------------------------------
    def save(self):
        meta = {"id": self.id, "name": self.name, "documents": self.documents, "chunks": self.chunks}
        self.meta_path.write_text(json.dumps(meta))
        if self.store is not None:
            self.store.save(self.index_path)

    @classmethod
    def load(cls, course_id: str) -> "Course":
        d = DATA_DIR / course_id
        meta = json.loads((d / "meta.json").read_text())
        course = cls(meta["id"], meta["name"])
        course.documents = meta["documents"]
        course.chunks = meta["chunks"]
        index_path = d / "index.faiss"
        if index_path.exists() and course.chunks:
            # dim is recoverable from a stored chunk's embedding call, but faster
            # to just read it back from the saved index itself.
            import faiss
            idx = faiss.read_index(str(index_path))
            course.store = VectorStore(dim=idx.d)
            course.store.index = idx
        return course


# --- a tiny on-disk registry so courses survive restarts -------------------

def create_course(name: str) -> Course:
    course_id = _slugify(name) + "-" + uuid.uuid4().hex[:6]
    course = Course(course_id, name)
    course.save()
    return course


def list_courses() -> List[Dict]:
    out = []
    if not DATA_DIR.exists():
        return out
    for d in sorted(DATA_DIR.iterdir()):
        meta_file = d / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            out.append({"id": meta["id"], "name": meta["name"], "documents": meta["documents"]})
    return out


def get_course(course_id: str) -> Course:
    if not (DATA_DIR / course_id / "meta.json").exists():
        raise FileNotFoundError(f"No course '{course_id}'")
    return Course.load(course_id)
