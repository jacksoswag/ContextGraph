from __future__ import annotations
import json, logging, os, re, time, hashlib, urllib.request
from typing import Callable

LOGGER = logging.getLogger(__name__)

_MODEL_TAGS: dict[str, str] = {
    "1B": os.getenv("DI_MODEL_1B", "llama3.2:1b"),
    "3B": os.getenv("DI_MODEL_3B", "llama3.2:3b"),
    "7B": os.getenv("DI_MODEL_7B", "llama3:8b"),
}
_OLLAMA = os.getenv("DI_OLLAMA_HOST", "http://localhost:11434")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_METRICS_PATH = os.getenv("DI_LLM_METRICS_PATH", ".di-ui/perf.jsonl")
_OLLAMA_DEFAULTS: dict = {"keep_alive": os.getenv("DI_LLM_KEEP_ALIVE", "10m")}

# Populated by ensure_ollama_models(). Monkeypatchable in tests.
_available: set[str] = set()

EventCallback = Callable[[str, dict], None] | None


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _record_call(payload: dict) -> None:
    if not _METRICS_PATH: return
    try:
        if d := os.path.dirname(_METRICS_PATH): os.makedirs(d, exist_ok=True)
        with open(_METRICS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception: pass


# Resolve tier string ("1B"/"3B"/"7B") → best available tag. Emits
# backend/tier_degraded when falling back. Monkeypatchable: tests set _available.
def _resolve_tag(tier: str, event_callback: EventCallback = None) -> str:
    requested_tag = _MODEL_TAGS.get(tier, _MODEL_TAGS["3B"])
    if not _available or requested_tag in _available: return requested_tag
    # Fallback cascade: 7B→3B→1B.
    for t in ("7B", "3B", "1B"):
        tag = _MODEL_TAGS[t]
        if tag in _available:
            if event_callback:
                try: event_callback("backend", {"event": "tier_degraded", "requested": tier, "actual_tag": tag})
                except Exception: pass
            return tag
    # Absolute last resort.
    tag = next(iter(_available))
    if event_callback:
        try: event_callback("backend", {"event": "tier_degraded", "requested": tier, "actual_tag": tag})
        except Exception: pass
    return tag


# Raw HTTP call — extracted for monkeypatching in tests.
# Signature: (prompt, tag, options, role, event_callback) → response text.
def _ollama_generate(prompt: str, tag: str, options: dict, role: str | None, event_callback: EventCallback) -> str:
    opts = dict(options)
    fmt = opts.pop("format", None)
    ka = opts.pop("keep_alive", None)
    payload: dict = {"model": tag, "prompt": prompt, "stream": False, "options": opts}
    if fmt is not None: payload["format"] = fmt
    if ka is not None: payload["keep_alive"] = ka

    t0 = time.perf_counter()
    err: str | None = None
    body: dict = {}
    text = ""
    try:
        req = urllib.request.Request(
            f"{_OLLAMA}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body.get("response", "") or ""
    except Exception as e:
        err = str(e)
        LOGGER.warning("Ollama connection failed: %s", e)

    wall_ms = (time.perf_counter() - t0) * 1000.0
    rec = {
        "ts": time.time(), "backend": "ollama", "role": role,
        "requested_tag": tag, "actual_tag": tag,
        "prompt_chars": len(prompt), "response_chars": len(text),
        "wall_ms": round(wall_ms, 2),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "eval_duration_ns": body.get("eval_duration"),
        "load_duration_ns": body.get("load_duration"),
        "error": err,
    }
    _record_call(rec)
    if event_callback:
        try: event_callback("llm_call", rec)
        except Exception: pass
    return text


def call_llm(prompt: str, model: str = "3B", *, options: dict | None = None,
             role: str | None = None, event_callback: EventCallback = None,
             prompt_cache_key: str | None = None) -> str:
    tag = _resolve_tag(model, event_callback)
    opts = {**_OLLAMA_DEFAULTS, **(options or {})}
    LOGGER.info("Calling Ollama (%s): %s...", tag, prompt[:50])
    return _ollama_generate(prompt, tag, opts, role, event_callback)


# JSON-seam cache: (model, options, prompt) → parsed dict. Seams are deterministic given inputs
# (temperature 0), so a repeated identical call must not re-hit the backend (spec §4).
_JSON_CACHE: dict[str, dict] = {}

def call_json(prompt: str, model: str = "3B", *, options: dict | None = None,
              role: str | None = None, event_callback: EventCallback = None,
              use_cache: bool = True) -> dict:
    opts = dict(options or {})
    opts.setdefault("temperature", 0)
    opts.setdefault("format", "json")
    key = hashlib.sha256(f"{model}|{json.dumps(opts, sort_keys=True)}|{prompt}".encode()).hexdigest()
    if use_cache and key in _JSON_CACHE: return _JSON_CACHE[key]
    raw = strip_think(call_llm(prompt, model, options=opts, role=role, event_callback=event_callback))
    result: dict = {}
    try: result = json.loads(raw)
    except Exception:
        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start != -1 and end > start: result = json.loads(raw[start:end])
        except Exception: result = {}
    if use_cache and result: _JSON_CACHE[key] = result   # never cache an empty/parse-failed result
    return result


# Pre-load tier models into Ollama's RAM with extended keep_alive so the
# first real pipeline call doesn't pay model-load latency (typically 1-5s
# per model, dominating short-prompt wall time). Sends a 1-token noop per
# tier; returns when all are resident. Safe to call repeatedly.
def warm_models(tiers=("1B", "3B", "7B"), *, keep_alive: str | None = None,
                event_callback: EventCallback = None) -> dict:
    ensure_ollama_models()
    out: dict[str, dict] = {}
    ka = keep_alive or _OLLAMA_DEFAULTS.get("keep_alive", "10m")
    for tier in tiers:
        tag = _resolve_tag(tier, event_callback)
        t0 = time.perf_counter()
        # num_predict=1 + format=json gives the most predictable cheap probe.
        _ollama_generate(
            prompt="warm", tag=tag,
            options={"num_predict": 1, "temperature": 0, "keep_alive": ka},
            role="warm", event_callback=event_callback,
        )
        out[tier] = {"tag": tag, "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    if event_callback:
        try: event_callback("warm_models", {"tiers": out})
        except Exception: pass
    return out


# Populate _available from the running Ollama instance.
def ensure_ollama_models() -> None:
    global _available
    try:
        req = urllib.request.Request(f"{_OLLAMA}/api/tags")
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.loads(r.read().decode("utf-8"))
        _available = {m["name"] for m in data.get("models", [])}
        LOGGER.info("Ollama models available: %s", _available)
    except Exception as exc:
        LOGGER.warning("Could not list Ollama models: %s", exc)
        _available = set(_MODEL_TAGS.values())
