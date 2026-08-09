"""
Central configuration for the Ramayana Knowledge Assistant.

All tunable constants live here so the rest of the backend (document
loading, chunking, embeddings, vector store, and the RAG pipeline)
can import from a single source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                 # put your Ramayana PDFs/DOCX/TXT here
PROMPTS_DIR = BASE_DIR / "prompts"            # persona system prompts
VECTOR_DB_ROOT = BASE_DIR / "vector_db"       # persisted FAISS indexes, one subfolder per provider

def vector_db_dir_for(provider: str) -> Path:
    """Each provider gets its own index folder — Gemini and OpenAI
    embeddings live in different vector spaces, so their indexes must
    never be mixed or loaded with the wrong embedding model."""
    return VECTOR_DB_ROOT / provider.lower()

# Kept for backward compatibility with any code still importing this directly.
VECTOR_DB_DIR = vector_db_dir_for("Gemini")

# ---------------------------------------------------------------------------
# AI Providers
# ---------------------------------------------------------------------------
# Set these as environment variables before running the app:
#   export GOOGLE_API_KEY="your-key-here"        (Linux/macOS)
#   setx GOOGLE_API_KEY "your-key-here"           (Windows)
# Never hardcode keys in source control.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Chat models available per provider, shown in the UI's model dropdown.
# The first entry in each list is the default.
AI_PROVIDERS = {
    "Gemini": {
        "chat_models": ["gemini-3.5-flash", "gemini-2.5-flash"],
        "embedding_model": "models/gemini-embedding-001",
        "key_placeholder": "AIza...",
        "get_key_url": "https://aistudio.google.com/apikey",
    },
    "OpenAI": {
        "chat_models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
        "embedding_model": "text-embedding-3-small",
        "key_placeholder": "sk-...",
        "get_key_url": "https://platform.openai.com/api-keys",
    },
}

# Kept for backward compatibility with any code still importing these directly.
GEMINI_CHAT_MODEL = AI_PROVIDERS["Gemini"]["chat_models"][0]
GEMINI_EMBEDDING_MODEL = AI_PROVIDERS["Gemini"]["embedding_model"]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 4          # number of chunks retrieved per query
SCORE_THRESHOLD = None     # optionally set a float (0-1) to filter weak matches

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
PERSONAS = {
    "Lord Rama": "rama.txt",
    "Lakshmana": "lakshmana.txt",
    "Hanuman": "hanuman.txt",
}

# Shown as clickable suggestions when a question can't be answered
# from the knowledge base — helps kids (or anyone) get back on track
# without needing to guess what to ask instead.
FALLBACK_QUESTIONS = {
    "Lord Rama": [
        "Why did you agree to go to the forest for 14 years?",
        "How did you meet Sita?",
        "What happened when you fought Ravana?",
    ],
    "Lakshmana": [
        "Why did you go with Rama into the forest?",
        "What is the Lakshmana Rekha?",
        "How did you help find Sita?",
    ],
    "Hanuman": [
        "How did you jump across the ocean to Lanka?",
        "How did you find Sita in Lanka?",
        "Why did you carry the whole mountain?",
    ],
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
TEMPERATURE = 0.4
MAX_OUTPUT_TOKENS = 2048

# ---------------------------------------------------------------------------
# Safety & abuse limits
# ---------------------------------------------------------------------------
MAX_QUESTION_LENGTH = 2000          # characters — rejects absurdly long input
MAX_AUDIO_SIZE_MB = 10              # rejects oversized voice recordings
RATE_LIMIT_MAX_REQUESTS = 10        # max questions...
RATE_LIMIT_WINDOW_SECONDS = 60      # ...per this many seconds, per browser session
CACHE_TTL_SECONDS = 3600            # how long a cached pipeline (and its API key) stays in server memory