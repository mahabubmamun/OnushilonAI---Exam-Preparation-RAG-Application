"""
Every knob for the pipeline lives here, read from environment variables
(loaded from .env). Mirrors the Kaggle notebook's config cell.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- LLM provider ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "ollama"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# --- Chunking ---
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", 450))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 80))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 5))

# --- Embeddings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# --- Generation ---
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 800))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

# --- Paths ---
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an AI exam preparation assistant.
Answer the student's question using ONLY the provided course materials.
If the answer cannot be found in the provided materials, say so clearly
instead of guessing. Cite the chunk numbers you used, like [Chunk 2].
Structure the answer clearly with headings or bullets where useful,
suitable for exam preparation."""
