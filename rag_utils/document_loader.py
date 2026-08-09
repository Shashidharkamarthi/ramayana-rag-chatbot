"""
Document loading and chunking utilities.

Reads Ramayana source documents (PDF, DOCX, TXT) from the data
directory, extracts their text, and splits them into overlapping
chunks ready for embedding.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR

# Map file extensions to their LangChain loader class.
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Load every supported file in `data_dir` into a list of LangChain
    Document objects. Each Document carries `metadata["source"]` set
    to the filename, which is later surfaced to users as a citation.

    Unsupported file types are skipped with a warning rather than
    raising, so one bad file doesn't block the whole ingestion run.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Create it and add your Ramayana PDF/DOCX/TXT files."
        )

    documents: List[Document] = []
    files = sorted(p for p in data_dir.iterdir() if p.is_file())

    if not files:
        raise ValueError(
            f"No files found in {data_dir}. Add at least one PDF, DOCX, "
            "or TXT Ramayana source document before building the index."
        )

    for file_path in files:
        suffix = file_path.suffix.lower()
        loader_cls = LOADER_MAP.get(suffix)

        if loader_cls is None:
            print(f"[skip] Unsupported file type, ignoring: {file_path.name}")
            continue

        try:
            if loader_cls is TextLoader:
                loader = loader_cls(str(file_path), encoding="utf-8")
            else:
                loader = loader_cls(str(file_path))
            loaded = loader.load()
        except Exception as exc:
            print(f"[error] Failed to load {file_path.name}: {exc}")
            continue

        # Normalize the source metadata to just the filename, so citations
        # shown to the user don't leak the full local filesystem path.
        for doc in loaded:
            doc.metadata["source"] = file_path.name

        documents.extend(loaded)
        print(f"[ok] Loaded {file_path.name} ({len(loaded)} page/section(s))")

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split loaded documents into overlapping chunks sized for embedding.

    Overlap preserves context across chunk boundaries so a sentence
    split mid-idea doesn't lose meaning during retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ok] Split {len(documents)} document section(s) into {len(chunks)} chunks")
    return chunks


def load_and_chunk(data_dir: Path = DATA_DIR) -> List[Document]:
    """Convenience wrapper: load all documents, then chunk them."""
    docs = load_documents(data_dir)
    return split_documents(docs)


if __name__ == "__main__":
    # Quick manual test: `python utils/document_loader.py`
    chunks = load_and_chunk()
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print("\n--- Sample chunk ---")
        print("Source:", chunks[0].metadata.get("source"))
        print(chunks[0].page_content[:300])