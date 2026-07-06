"""
Document Ingestion Module
--------------------------
Loads raw text out of PDFs / .txt files so the rest of the pipeline
never has to care what format a document arrived in.

No external API calls are made anywhere in this module.
"""

from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_pdf(path: str) -> str:
    """Extract raw text from every page of a PDF."""
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def load_text_file(path: str) -> str:
    """Read a plain text / markdown file."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_document(path: str) -> Dict:
    """
    Load a single document (PDF or text) and return a dict with
    the raw text plus light metadata used later for citations.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {SUPPORTED_EXTENSIONS}"
        )

    if ext == ".pdf":
        raw_text = load_pdf(path)
    else:
        raw_text = load_text_file(path)

    # Basic cleanup: collapse excessive whitespace/newlines
    cleaned = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())

    return {
        "source": p.name,
        "path": str(p),
        "num_chars": len(cleaned),
        "text": cleaned,
    }


def load_documents(paths: List[str]) -> List[Dict]:
    """Load multiple documents, skipping any that fail, and report why."""
    documents = []
    for path in paths:
        try:
            documents.append(load_document(path))
        except Exception as exc:
            print(f"[ingestion] Skipped '{path}': {exc}")
    return documents
