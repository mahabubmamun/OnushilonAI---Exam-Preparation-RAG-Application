"""
PDF text extraction — keeping page numbers, so citations are possible later.
Same logic as the Kaggle notebook, PyMuPDF only (no pypdf fallback needed
locally since PyMuPDF installs cleanly outside Kaggle).
"""
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF


def extract_pages(pdf_path: str) -> List[Dict]:
    """Extract text page by page. Returns [{'page': 1, 'text': '...'}, ...]."""
    pages = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    finally:
        doc.close()
    return pages
