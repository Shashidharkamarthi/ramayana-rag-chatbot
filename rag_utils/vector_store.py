"""
Vector store utilities.

Builds a FAISS index from document chunks, using either Gemini or
OpenAI embeddings depending on the selected provider, and provides
save/load helpers so the index only needs to be built once per
provider and can be reused across app restarts.
"""

import hashlib
import sys
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AI_PROVIDERS, GOOGLE_API_KEY, OPENAI_API_KEY, vector_db_dir_for

# Gemini's free tier caps embedding calls at 100/minute. Batch chunks
# well under that limit and pace the batches so building a large index
# never hits a 429 RESOURCE_EXHAUSTED error partway through.
EMBED_BATCH_SIZE = 80
EMBED_BATCH_PAUSE_SECONDS = 61

MANIFEST_FILENAME = "manifest.sha256"


def _compute_index_hash(load_path: Path) -> str:
    """Hash of the FAISS index files, used to detect tampering."""
    hasher = hashlib.sha256()
    for fname in ("index.faiss", "index.pkl"):
        with open(load_path / fname, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def _write_manifest(save_path: Path) -> None:
    """Record a hash of the index we just built ourselves, so a future
    load can verify the files haven't been swapped or tampered with."""
    digest = _compute_index_hash(save_path)
    (save_path / MANIFEST_FILENAME).write_text(digest)


def _verify_manifest(load_path: Path) -> bool:
    """
    Confirm the index on disk still matches the hash recorded when we
    built it. FAISS.load_local unpickles index.pkl, and unpickling
    arbitrary/tampered data can execute code — this check ensures we
    only ever deserialize a file we know we wrote ourselves.
    """
    manifest_file = load_path / MANIFEST_FILENAME
    if not manifest_file.exists():
        return False
    expected = manifest_file.read_text().strip()
    actual = _compute_index_hash(load_path)
    return expected == actual


def get_embedding_model(provider: str = "Gemini", api_key: Optional[str] = None):
    """
    Construct the embedding model for the given provider, used for
    both indexing document chunks and embedding user queries at
    retrieval time. Both sides must use the same provider/model or
    similarity scores are meaningless.

    `api_key` lets a caller (e.g. a per-user key entered in the UI)
    override the app's default key for that provider.
    """
    if provider == "Gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        effective_key = api_key or GOOGLE_API_KEY
        if not effective_key:
            raise EnvironmentError(
                "No Gemini API key available. Either export GOOGLE_API_KEY, "
                "or provide one at runtime (e.g. via the app's sidebar)."
            )
        return GoogleGenerativeAIEmbeddings(
            model=AI_PROVIDERS["Gemini"]["embedding_model"],
            google_api_key=effective_key,
        )

    elif provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        effective_key = api_key or OPENAI_API_KEY
        if not effective_key:
            raise EnvironmentError(
                "No OpenAI API key available. Either export OPENAI_API_KEY, "
                "or provide one at runtime (e.g. via the app's sidebar)."
            )
        return OpenAIEmbeddings(
            model=AI_PROVIDERS["OpenAI"]["embedding_model"],
            api_key=effective_key,
        )

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Expected 'Gemini' or 'OpenAI'.")


def build_vector_store(
    chunks: List[Document],
    provider: str = "Gemini",
    api_key: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> FAISS:
    """
    Embed all chunks and build a new FAISS index from scratch, then
    persist it to disk so it can be reloaded without re-embedding.

    Chunks are embedded in paced batches (not all at once) to stay
    under the free-tier rate limit — see EMBED_BATCH_SIZE above.
    For a few thousand chunks this can take a while on the free tier;
    it only needs to happen once per provider, since the result is
    saved to disk and reloaded instantly on every future run.
    """
    if not chunks:
        raise ValueError("No chunks provided — nothing to index.")

    save_path = Path(save_path) if save_path else vector_db_dir_for(provider)

    embeddings = get_embedding_model(provider=provider, api_key=api_key)
    embedding_name = AI_PROVIDERS[provider]["embedding_model"]

    total = len(chunks)
    total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    print(f"[ok] Embedding {total} chunks with {provider} ({embedding_name}) "
          f"in {total_batches} batch(es)...")

    vector_store: Optional[FAISS] = None
    for batch_num, start in enumerate(range(0, total, EMBED_BATCH_SIZE), start=1):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        print(f"[ok] Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

        if batch_num < total_batches:
            print(f"[ok] Pausing {EMBED_BATCH_PAUSE_SECONDS}s to stay under the "
                  f"free-tier rate limit...")
            time.sleep(EMBED_BATCH_PAUSE_SECONDS)

    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))
    _write_manifest(save_path)
    print(f"[ok] FAISS index saved to {save_path}")

    return vector_store


def load_vector_store(
    provider: str = "Gemini",
    api_key: Optional[str] = None,
    load_path: Optional[Path] = None,
) -> Optional[FAISS]:
    """
    Load a previously built FAISS index for this provider from disk.
    Returns None if no index exists yet at that path, so callers can
    decide whether to trigger ingestion instead of crashing.
    """
    load_path = Path(load_path) if load_path else vector_db_dir_for(provider)
    index_file = load_path / "index.faiss"

    if not index_file.exists():
        return None

    if not _verify_manifest(load_path):
        raise RuntimeError(
            f"Vector store integrity check failed at {load_path} — the index "
            "files don't match their recorded hash, which could mean they "
            "were modified or corrupted. Refusing to load. Delete this "
            "folder and let the app rebuild it to fix this."
        )

    embeddings = get_embedding_model(provider=provider, api_key=api_key)
    vector_store = FAISS.load_local(
        str(load_path),
        embeddings,
        # Safe here specifically because _verify_manifest() above just
        # confirmed these files match the hash we recorded when WE built
        # them — we never load a FAISS index we didn't create ourselves.
        allow_dangerous_deserialization=True,
    )
    print(f"[ok] FAISS index loaded from {load_path}")
    return vector_store


def get_or_build_vector_store(
    data_dir: Optional[Path] = None,
    force_rebuild: bool = False,
    provider: str = "Gemini",
    api_key: Optional[str] = None,
) -> FAISS:
    """
    Main entry point for the app: load this provider's existing index
    if present, otherwise build one from the documents in `data_dir`.
    Each provider keeps its own index folder, so switching providers
    in the UI safely triggers a one-time rebuild instead of ever
    mixing incompatible embedding spaces.
    """
    if not force_rebuild:
        existing = load_vector_store(provider=provider, api_key=api_key)
        if existing is not None:
            return existing

    # Local import avoids a circular import at module load time.
    from rag_utils.document_loader import load_and_chunk

    chunks = load_and_chunk(data_dir) if data_dir else load_and_chunk()
    return build_vector_store(chunks, provider=provider, api_key=api_key)


if __name__ == "__main__":
    # Quick manual test: `python rag_utils/vector_store.py`
    store = get_or_build_vector_store(force_rebuild=True)
    results = store.similarity_search("Why did Rama accept exile?", k=3)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:200])