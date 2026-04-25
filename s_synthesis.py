import json, os, re, requests
from utils import display_source
from constants import (
    DEFAULT_OLLAMA_CONNECT_TIMEOUT,
    DEFAULT_OLLAMA_READ_TIMEOUT,
    OLLAMA_GENERATE_URL,
    SYNTHESIS_MAX_CHAIN_CLAUSES,
    SYNTHESIS_MODEL,
    SYNTHESIS_TARGET_MATCH_THRESHOLD,
)

SPECIFIC_CITATION_RE = re.compile(r"(\d|%|\b[A-Z]{2,}\b)")
SOURCE_LINE_RE = re.compile(r"^\s*\[\d+\]\s*:?.*$")
SOURCE_REF_RE = re.compile(r"\[(\d+)\]")
TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")

class KnowledgeSynthesizer: # Synthesizes arguments into a report using Ollama 3B
    def __init__(self, ollama_url=OLLAMA_GENERATE_URL, model=SYNTHESIS_MODEL):
        self.ollama_url = ollama_url
        self.model = model
        self.connect_timeout = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", DEFAULT_OLLAMA_CONNECT_TIMEOUT))
        self.read_timeout = float(os.getenv("OLLAMA_READ_TIMEOUT", DEFAULT_OLLAMA_READ_TIMEOUT))

    def _should_cite_clause(self, clause) -> bool:
        return bool(SPECIFIC_CITATION_RE.search(str(clause or "")))

    def _truncate_structured_chain(self, chain, max_steps=20):
        if not chain or len(chain) <= max_steps + 1:
            return chain
        return [chain[0]] + list(chain[-max_steps:])

    def _clause_text(self, clause) -> str:
        text = " ".join(str(clause or "").strip().split())
        return f"{text.rstrip('.')}." if text else ""

    def _temporal_context_text(self, temporal) -> str:
        if not isinstance(temporal, dict):
            return ""
        text = " ".join(str(temporal.get("text") or "").strip().split())
        if text:
            return text.rstrip(".") + "."
        original = str(temporal.get("original_tense") or "").strip()
        current = str(temporal.get("current_tense") or "").strip()
        confidence = str(temporal.get("temporal_confidence") or "").strip()
        if not any((original, current, confidence)):
            return ""
        return (
            f"Original source-time tense: {original or 'unknown'}; "
            f"current-relative tense: {current or 'unknown'}; "
            f"temporal confidence: {confidence or 'unknown'}."
        )

    def _step_clause(self, step) -> str:
        if not isinstance(step, dict):
            return ""
        return self._clause_text(step.get("evidence_text", ""))

    def _chain_records(self, chain):
        records = []
        if not chain or not isinstance(chain[0], dict):
            return records
        chain = self._truncate_structured_chain(chain)
        for step in chain[1:]:
            clause = self._step_clause(step)
            if clause:
                records.append(
                    {
                        "text": clause,
                        "source": str(step.get("source") or "").strip(),
                        "role": "path",
                        "temporal": dict(step.get("temporal") or {}),
                    }
                )
            for detail in list(step.get("specific_details", []) or []):
                if not isinstance(detail, dict):
                    continue
                text = self._clause_text(detail.get("text", ""))
                if not text:
                    continue
                records.append(
                    {
                        "text": text,
                        "source": str(detail.get("source") or "").strip(),
                        "role": "specific_context",
                        "temporal": dict(detail.get("temporal") or {}),
                    }
                )
        return records

    def _format_argument_notes(self, chain, source_map=None) -> str:
        del source_map
        if not chain or not isinstance(chain[0], dict):
            return self._format_argument(chain)

        return self._record_chain_text(self._chain_records(chain))

    def _format_argument(self, chain, source_map=None) -> str:
        del source_map
        if isinstance(chain, str):
            return chain.strip()
        if not chain:
            return ""

        if isinstance(chain[0], dict):
            return self._record_chain_text(self._chain_records(chain))

        return self._sentence_chain_text(chain)

    def _source_refs(self, sources, source_map) -> str:
        return " ".join(
            f"[{source_map[display_source(source)]}]"
            for source in list(sources or [])
            if display_source(source) in source_map
        )

    def _record_refs(self, record, source_map) -> str:
        source = display_source((record or {}).get("source"))
        if not source or source not in source_map:
            return ""
        return f"[{source_map[source]}]"

    def _sentence_chain_text(self, clauses, sources=None, source_map=None) -> str:
        source_refs = self._source_refs(sources, source_map or {})
        sentences = []
        for clause in list(clauses or []):
            text = self._clause_text(clause)
            if not text:
                continue
            if source_refs and self._should_cite_clause(text):
                text = f"{text.rstrip('.')} {source_refs}."
            sentences.append(text)
        return " ".join(sentences)

    def _record_chain_text(self, records, source_map=None) -> str:
        sentences = []
        for record in list(records or []):
            text = self._clause_text((record or {}).get("text", ""))
            if not text:
                continue
            source = str((record or {}).get("source") or "").strip()
            temporal_text = self._temporal_context_text((record or {}).get("temporal"))
            if temporal_text:
                text = f"{text.rstrip('.')} (Temporal context: {temporal_text.rstrip('.')})."
            source_refs = self._source_refs([source], source_map or {})
            if source_refs:
                text = f"{text.rstrip('.')} {source_refs}."
            sentences.append(text)
        return " ".join(sentences)

    def _payload_records(self, payload):
        records = []
        for record in list((payload or {}).get("clause_records", []) or []):
            if not isinstance(record, dict):
                continue
            text = self._clause_text(record.get("text", ""))
            if not text:
                continue
            records.append(
                {
                    "text": text,
                    "source": str(record.get("source") or "").strip(),
                    "role": str(record.get("role") or "path").strip() or "path",
                    "temporal": dict(record.get("temporal") or {}),
                }
            )

        if records:
            return records

        sources = [
            str(source or "").strip()
            for source in list((payload or {}).get("sources", []) or [])
            if str(source or "").strip() and str(source or "").strip() != "unknown"
        ]
        fallback_source = sources[0] if len(sources) == 1 else ""
        for clause in list((payload or {}).get("path_clauses", []) or []):
            text = self._clause_text(clause)
            if text:
                records.append({"text": text, "source": fallback_source, "role": "path", "temporal": {}})
        for clause in list((payload or {}).get("supporting_clauses", []) or []):
            text = self._clause_text(clause)
            if text:
                records.append({"text": text, "source": fallback_source, "role": "specific_context", "temporal": {}})
        return records

    def _build_source_map(self, payloads) -> dict:
        source_map = {}
        for payload in payloads:
            for record in self._payload_records(payload):
                source = display_source((record or {}).get("source"))
                if source and source != "unknown" and source not in source_map:
                    source_map[source] = len(source_map) + 1
        return source_map

    def _strip_existing_source_list(self, text):
        kept = []
        for line in str(text or "").splitlines():
            clean = line.strip()
            heading_text = clean.lstrip("#").strip().lower()
            if heading_text in {"sources", "source list", "references"}:
                continue
            if SOURCE_LINE_RE.match(clean):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def _normalize_report_heading(self, text):
        lines = str(text or "").strip().splitlines()
        for index, line in enumerate(lines):
            clean = line.strip()
            if not clean:
                continue

            heading = re.match(r"^(#{1,4})\s+(.+)$", clean)
            if heading:
                title = self._strip_terminal_source_refs(heading.group(2)).rstrip(".").strip()
                if title:
                    lines[index] = f"## {title}"
                break

            next_is_blank = index + 1 < len(lines) and not lines[index + 1].strip()
            title = self._strip_terminal_source_refs(clean).rstrip(".").strip()
            title_word_count = len(TEXT_TOKEN_RE.findall(title))
            if next_is_blank and 2 <= title_word_count <= 22:
                lines[index] = f"## {title}"
            break
        return "\n".join(lines).strip()

    def _sentence_has_source_ref(self, sentence):
        return bool(re.search(r"\[\d+\](?:\s*\[\d+\])*\s*[.!?]?$", sentence.strip()))

    def _strip_terminal_source_refs(self, sentence):
        sentence = sentence.strip()
        match = re.search(r"\s*(?:\[\d+\]\s*)+([.!?])?\s*$", sentence)
        if not match:
            return sentence
        punctuation = match.group(1) or ""
        stripped = sentence[:match.start()].rstrip()
        if punctuation and stripped[-1:] not in ".!?":
            stripped = f"{stripped}{punctuation}"
        return stripped

    def _citation_tokens(self, text):
        return {
            token
            for token in TEXT_TOKEN_RE.findall(str(text or "").lower())
            if len(token) > 2
        }

    def _best_evidence_refs(self, sentence, records, source_map, limit=2):
        sentence_tokens = self._citation_tokens(sentence)
        if not sentence_tokens:
            return ""

        ranked = []
        seen_refs = set()
        for record in list(records or []):
            ref = self._record_refs(record, source_map)
            if not ref or ref in seen_refs:
                continue
            record_tokens = self._citation_tokens((record or {}).get("text", ""))
            if not record_tokens:
                continue
            overlap = sentence_tokens & record_tokens
            if len(overlap) < 2:
                continue
            precision = len(overlap) / len(sentence_tokens)
            recall = len(overlap) / len(record_tokens)
            score = max(precision, 0.85 * recall)
            if score < 0.22:
                continue
            ranked.append((score, len(overlap), ref))
            seen_refs.add(ref)

        ranked.sort(reverse=True)
        return " ".join(ref for _score, _overlap, ref in ranked[:limit])

    def _append_ref_to_sentence(self, sentence, ref):
        sentence = sentence.strip()
        if not sentence or self._sentence_has_source_ref(sentence):
            return sentence
        if sentence[-1:] in ".!?":
            return f"{sentence[:-1].rstrip()} {ref}{sentence[-1]}"
        return f"{sentence} {ref}."

    def _ensure_sentence_citations(self, text, source_map, records=None):
        if not source_map:
            return str(text or "").strip()
        paragraphs = []
        for paragraph in str(text or "").split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if paragraph.lstrip().startswith("#"):
                paragraphs.append(paragraph)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            cited_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                base_sentence = self._strip_terminal_source_refs(sentence)
                refs = self._best_evidence_refs(base_sentence, records, source_map)
                if refs:
                    cited_sentences.append(self._append_ref_to_sentence(base_sentence, refs))
                    continue
                cited_sentences.append(
                    base_sentence or sentence
                )
            paragraphs.append(" ".join(cited_sentences))
        return "\n\n".join(paragraphs).strip()

    def _append_source_list(self, text, source_map):
        if not source_map:
            return str(text or "").strip()
        source_lines = "\n".join(
            f"[{idx}]: {source}"
            for source, idx in sorted(source_map.items(), key=lambda item: item[1])
        )
        return f"{str(text or '').strip()}\n\n## Sources\n{source_lines}".strip()

    def _compact_source_map(self, text, source_map):
        reverse_map = {idx: source for source, idx in source_map.items()}
        old_to_new = {}
        compact_source_map = {}

        for match in SOURCE_REF_RE.finditer(str(text or "")):
            old_idx = int(match.group(1))
            source = reverse_map.get(old_idx)
            if not source or old_idx in old_to_new:
                continue
            new_idx = len(old_to_new) + 1
            old_to_new[old_idx] = new_idx
            compact_source_map[source] = new_idx

        if not old_to_new:
            return str(text or "").strip(), {}

        def replace_ref(match):
            old_idx = int(match.group(1))
            new_idx = old_to_new.get(old_idx)
            return f"[{new_idx}]" if new_idx is not None else ""

        compact_text = SOURCE_REF_RE.sub(replace_ref, str(text or "")).strip()
        compact_text = re.sub(r"\s+([.!?])", r"\1", compact_text)
        compact_text = re.sub(r"[ \t]{2,}", " ", compact_text)
        return compact_text, compact_source_map

    def synthesize_relationship_report(self, target_a: str, target_b: str, payloads: list[dict]) -> str:
        payloads = [dict(item or {}) for item in list(payloads or []) if isinstance(item, dict)]
        payloads = [
            payload
            for payload in payloads
            if len(list(payload.get("path_clauses", []) or [])) >= 2
            and float(payload.get("match_target_a", 0.0) or 0.0) >= SYNTHESIS_TARGET_MATCH_THRESHOLD
            and float(payload.get("match_target_b", 0.0) or 0.0) >= SYNTHESIS_TARGET_MATCH_THRESHOLD
        ]
        if not payloads:
            return f"No successful relationship chains were found between {target_a or 'target A'} and {target_b or 'target B'}."

        source_map = self._build_source_map(payloads)

        note_blocks = []
        evidence_records = []
        for payload in payloads:
            records = self._payload_records(payload)
            evidence_records.extend(records)
            path_records = [
                record for record in records if record.get("role") == "path"
            ][:SYNTHESIS_MAX_CHAIN_CLAUSES]
            context_records = [
                record for record in records if record.get("role") == "specific_context"
            ][:SYNTHESIS_MAX_CHAIN_CLAUSES * 3]
            path_text = self._record_chain_text(path_records, source_map=source_map)
            context_text = self._record_chain_text(context_records, source_map=source_map)
            if not path_text:
                continue
            block_parts = [f"Relationship evidence: {path_text}"]
            if context_text:
                block_parts.append(f"Concrete details: {context_text}")
            note_blocks.append("\n".join(block_parts))

        digest = "\n\n".join(note_blocks)
        source_lines = "\n".join(
            f"[{idx}]: {source}"
            for source, idx in sorted(source_map.items(), key=lambda item: item[1])
        )

        prompt = f"""
            Write a research report about the connections between:
            Target A: "{target_a}"
            Target B: "{target_b}"

            Use ONLY the evidence below.

            Evidence:
            {digest}

            Source index:
            {source_lines or "(no explicit sources available)"}

            Requirements:
            - Use markdown: start with one neutral `##` title and use `##` section headings if helpful.
            - Do not put source markers on titles or headings.
            - Write directly about Target A, Target B, and the substantive relationship between them.
            - Do not describe your method, the input format, or the fact that evidence was provided.
            - Treat relationship evidence as the core sequence and concrete details as supporting facts about entities in that sequence.
            - Each paragraph should be one substantive argument grounded in the provided wording.
            - Use only the provided wording to draw conclusions.
            - Report on how Target A and Target B are connected only when the provided wording actually supports that link.
            - If evidence only shows general expertise, revenue, scale, authority, or global presence, do not infer specific involvement in Target B.
            - Do not use examples from unrelated places, programs, or organizations as support for Target B unless the evidence explicitly links them to Target B.
            - If evidence separately describes Target A and Target B but does not bridge them, say the relationship is insufficiently supported.
            - Match the title and conclusion to the actual strength of the evidence: direct, indirect, weak, or insufficient.
            - Cite each source-grounded sentence with the marker already attached to the evidence it uses, like [1] or [1] [2].
            - Do not reuse a source marker unless that sentence relies on that source's evidence.
            - Use Temporal context to distinguish source-time wording from current-time wording.
            - If Temporal context says an explicit event date resolves as past/present/future today, write the claim in that current-relative tense even if the original clause says "will" or "is".
            - If Temporal context says a present/future clause is stale, unknown, or projection-only, phrase it as source-time description or projection, not as a confirmed current fact.
            - Preserve concrete figures, dates, proper nouns, and source-linked claims exactly as written in the clauses.
            - Write in concise formal prose.
            - If the chains are weak, indirect, or insufficient, say that plainly instead of forcing a strong conclusion.
            - Do not invent sources, mechanisms, causal claims, or agreement levels that are not in the records.
            - Do not add notes, implementation comments, or commentary about the report itself.
            - End with a short conclusion about the strongest supported link between Target A and Target B, or state that support is insufficient.

            Return only the finished report text.
        """

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.5,
                },
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                stream=True,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            response.raise_for_status()

            chunks = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    chunks.append(chunk)
                if data.get("done"):
                    break

            final_text = "".join(chunks).strip()
            if not final_text:
                return "Error: No response from model."
            final_text = self._strip_existing_source_list(final_text)
            final_text = self._normalize_report_heading(final_text)
            final_text = self._ensure_sentence_citations(final_text, source_map, evidence_records)
            final_text, used_source_map = self._compact_source_map(final_text, source_map)
            return self._append_source_list(final_text, used_source_map)

        except requests.exceptions.ConnectionError:
            return f"Synthesis Failed: Could not connect to Ollama at {self.ollama_url}. Start Ollama and retry."
        except requests.exceptions.ReadTimeout:
            return f"Synthesis Failed: Ollama did not finish within {int(self.read_timeout)} seconds. Increase OLLAMA_READ_TIMEOUT or use a smaller/faster model."
        except Exception as e:
            return f"Synthesis Failed: {str(e)}"
