from __future__ import annotations
from pathlib import Path
from typing import Iterator
from ingest.extraction import extract_clauses
from ingest.file_reformatter import reformat_file

_SUPPORTED = {".txt", ".md"}

# Produce typed-triple dicts from a .txt/.md file. Pre-curated file content
# bypasses the bare-question filter; reformatting strips markdown to plain
# segments before SVO extraction.
def ingest_file(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if path.suffix.lower() not in _SUPPORTED: raise ValueError(f"unsupported file type: {path.suffix}")
    yield from ingest_text_file(path.read_text(encoding="utf-8", errors="replace"))

def ingest_text_file(text: str) -> Iterator[dict]:
    for segment in reformat_file(text):
        yield from extract_clauses(segment)
