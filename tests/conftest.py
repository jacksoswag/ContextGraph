from __future__ import annotations
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Isolation: redirect all on-disk cache paths to tmp_path so no test can touch .di-ui/.
@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    try:
        import ingest.scrape_worker as sw_mod
        import ingest.deep_ingest as di_mod
    except ModuleNotFoundError:
        return  # modules not importable in this env; skip path isolation

    fake_rc   = tmp_path / "research_cache.sqlite"
    fake_perf = tmp_path / "perf.jsonl"

    monkeypatch.setenv("RESEARCH_CACHE_PATH", str(fake_rc))
    monkeypatch.setenv("DI_LLM_METRICS_PATH", str(fake_perf))
    monkeypatch.setattr(di_mod, "_CACHE_PATH", str(fake_rc))
    monkeypatch.setattr(sw_mod, "RESEARCH_CACHE_PATH", fake_rc)


# Deterministic mock LLM: build a (prompt, model) → dict callable from a responder, recording every
# call on `.calls`. Contract-tests the LLM seams (and the e2e tree) with no backend.
@pytest.fixture
def mock_llm():
    def make(responder):
        calls: list[tuple[str, str]] = []
        def llm(prompt: str, model: str = "3B") -> dict:
            calls.append((prompt, model))
            return responder(prompt, model)
        llm.calls = calls
        return llm
    return make
