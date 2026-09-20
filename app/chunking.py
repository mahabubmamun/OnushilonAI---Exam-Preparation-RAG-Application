"""
Word-based chunking with overlap. Each chunk remembers the page it
started on -> that becomes the citation. Identical logic to the notebook.
"""
from typing import List, Dict
from app.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def chunk_pages(pages: List[Dict],
                 chunk_size: int = CHUNK_SIZE_WORDS,
                 overlap: int = CHUNK_OVERLAP_WORDS) -> List[Dict]:
    """Turn page-level text into overlapping word chunks that remember their page."""
    chunks, buf_words, buf_pages = [], [], []

    def flush():
        if buf_words:
            chunks.append({
                "text": " ".join(buf_words),
                "page": buf_pages[0],  # page this chunk starts on
            })

    for page_data in pages:
        words = page_data["text"].split()
        idx = 0
        while idx < len(words):
            space_left = chunk_size - len(buf_words)
            take = words[idx: idx + space_left]
            buf_words.extend(take)
            buf_pages.extend([page_data["page"]] * len(take))
            idx += len(take)

            if len(buf_words) >= chunk_size:
                flush()
                if overlap > 0:
                    buf_words, buf_pages = buf_words[-overlap:], buf_pages[-overlap:]
                else:
                    buf_words, buf_pages = [], []

    flush()
    return chunks
