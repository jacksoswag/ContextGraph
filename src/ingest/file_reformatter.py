from __future__ import annotations
import re

_CODE_FENCE = re.compile(r"^\s*```")
_HEADER     = re.compile(r"^#{1,6}\s+")
_BULLET     = re.compile(r"^\s*[-*+]\s+|^\s*\d+[.)]\s+")
_HRULE      = re.compile(r"^\s*[-*=_]{3,}\s*$")

def _strip_md(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", line)              # images
    line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", line)          # links → text
    line = re.sub(r"`+([^`]*)`+", r"\1", line)                     # inline code
    line = re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", line)           # bold/italic *
    line = re.sub(r"_{1,3}([^_]*)_{1,3}", r"\1", line)             # bold/italic _
    line = re.sub(r"~~([^~]*)~~", r"\1", line)                     # strikethrough
    line = re.sub(r"<[^>]+>", "", line)                            # HTML tags
    return line.strip()

# Parse markdown/text into clean declarative segments ready for extract_clauses().
# Strips structural markup, skips code blocks and rules, preserves imperative lines.
def reformat_file(text: str) -> list[str]:
    in_code, out = False, []
    for raw in text.splitlines():
        if _CODE_FENCE.match(raw): in_code = not in_code; continue
        if in_code or _HRULE.match(raw) or raw.strip().startswith("<!--"): continue
        line = _HEADER.sub("", raw)
        line = _BULLET.sub("", line)
        line = _strip_md(line)
        if line: out.append(line)
    return out
