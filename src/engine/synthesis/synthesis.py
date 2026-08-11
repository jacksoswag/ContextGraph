import json, os, re, requests; from engine.common.shm import display_source; from engine.extract.noise_cleanup import is_usable_clause_text; from engine.common.constants import (DEFAULT_OLLAMA_CONNECT_TIMEOUT, DEFAULT_OLLAMA_READ_TIMEOUT, OLLAMA_GENERATE_URL, SYNTHESIS_ARGUMENT_LIMIT, SYNTHESIS_MAX_CHAIN_CLAUSES, SYNTHESIS_MODEL, SYNTHESIS_TARGET_MATCH_THRESHOLD,)
from engine.extract.target_text import target_tokens; SPECIFIC_CITATION_RE = re.compile(r"(\d|%|\b[A-Z]{2,}\b)"); CONCRETE_DETAIL_RE = re.compile(
    r"(\$|%|\b\d[\d,./:-]*\b|\b(?:million|billion|trillion|percent|dollars?|median|rent|price|cost|income)\b)", re.I,
); SOURCE_LINE_RE = re.compile(r"^\s*\[\d+\]\s*:?.*$"); SOURCE_REF_RE = re.compile(r"\[(\d+)\]"); TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+"); PAYLOAD_DIVERSITY_THRESHOLD = 0.72
TEMPLATE_REPORT_HEADINGS = {"abstract", "analysis", "background", "conclusion", "concrete details", "concrete details continued", "discussion", "evidence", "findings", "introduction", "limitations", "relationship evidence", "supporting evidence", "the relationship between target a and target b",}; PIPELINE_LANGUAGE_REPLACEMENTS = (
    (re.compile(r"\bprovided chain\b", re.IGNORECASE), "cited facts"), (re.compile(r"\bprovided chains\b", re.IGNORECASE), "cited facts"), (re.compile(r"\bargument chain\b", re.IGNORECASE), "relationship"), (re.compile(r"\bargument chains\b", re.IGNORECASE), "relationships"), (re.compile(r"\bevidence block\b", re.IGNORECASE), "cited facts"),
    (re.compile(r"\bevidence blocks\b", re.IGNORECASE), "cited facts"), (re.compile(r"\bpayload\b", re.IGNORECASE), "claim"), (re.compile(r"\brecord\b", re.IGNORECASE), "source detail"), (re.compile(r"\bextracted graph\b", re.IGNORECASE), "cited facts"),
); EVIDENCE_REPORTING_RE = re.compile(
    r"\b(?:the\s+)?(?:(?:available|provided|cited)\s+)?evidence\s+" r"(?:suggests|shows|indicates|demonstrates|supports|implies)\s+(?:that\s+)?", re.IGNORECASE,
); FACT_REPORTING_RE = re.compile(
    r"\b(?:the\s+)?(?:(?:available|provided|cited)\s+)?facts\s+" r"(?:suggest|show|indicate|demonstrate|support|imply)\s+(?:that\s+)?", re.IGNORECASE,
)
# Turns successful thought payloads into a source-cited relationship report.
class KnowledgeSynthesizer: # Synthesizes arguments into a report using Ollama 3B
    # Initializes this object from caller-provided state.
    def __init__(self, ollama_url=OLLAMA_GENERATE_URL, model=SYNTHESIS_MODEL):
        self.ollama_url = ollama_url; self.model = model; self.connect_timeout = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", DEFAULT_OLLAMA_CONNECT_TIMEOUT)); self.read_timeout = float(os.getenv("OLLAMA_READ_TIMEOUT", DEFAULT_OLLAMA_READ_TIMEOUT))
    # Returns whether a clause contains concrete detail worth citing.
    def _should_cite_clause(self, clause) -> bool:
        return bool(SPECIFIC_CITATION_RE.search(str(clause or "")))
    # Truncates truncate structured chain for final synthesis.
    def _truncate_structured_chain(self, chain, max_steps=20):
        if not chain or len(chain) <= max_steps + 1:
            return chain
        return [chain[0]] + list(chain[-max_steps:])
    # Returns clause text for final synthesis.
    def _clause_text(self, clause) -> str:
        text = " ".join(str(clause or "").strip().split())
        return f"{text.rstrip('.')}." if text else ""
    # Returns step clause for final synthesis.
    def _step_clause(self, step) -> str:
        if not isinstance(step, dict):
            return ""
        return self._clause_text(step.get("evidence_text", ""))
    # Returns whether source is citable for final synthesis.
    def _source_is_citable(self, source) -> bool:
        raw = " ".join(str(source or "").strip().split()); display = display_source(raw)
        return bool(display and display != "unknown" and ("|" in raw or "://" in display or "/" in display or "." in display))
    # Returns record key for final synthesis.
    def _record_key(self, record, include_role=True):
        text = self._clause_text((record or {}).get("text", "")).lower(); text = " ".join(TEXT_TOKEN_RE.findall(text)); source = display_source((record or {}).get("source", "")).rstrip("/").lower()
        if include_role:
            return (str((record or {}).get("role") or "path").strip() or "path", text, source)
        return (text, source)
    # Returns record claim tokens for final synthesis.
    def _record_claim_tokens(self, record):
        text = re.sub(r"\[\d+\]", "", str((record or {}).get("text") or ""))
        return set(target_tokens(text, min_length=3))
    # Returns whether claim-token overlap makes a record duplicate.
    def _claim_token_duplicate(self, tokens, seen_token_sets, overlap_threshold=0.72, jaccard_threshold=0.60):
        tokens = set(tokens or [])
        if len(tokens) < 4:
            return False
        for seen_tokens in list(seen_token_sets or []):
            seen_tokens = set(seen_tokens or [])
            if len(seen_tokens) < 4:
                continue
            overlap = len(tokens & seen_tokens)
            if not overlap:
                continue
            if overlap / max(1, min(len(tokens), len(seen_tokens))) >= overlap_threshold:
                return True
            if overlap / max(1, len(tokens | seen_tokens)) >= jaccard_threshold:
                return True
        return False
    # Returns dedupe similar records used by final synthesis.
    def _dedupe_similar_records(self, records, seen_token_sets=None):
        deduped = []; seen_token_sets = list(seen_token_sets or [])
        for record in list(records or []):
            tokens = self._record_claim_tokens(record)
            if self._claim_token_duplicate(tokens, seen_token_sets):
                continue
            deduped.append(record)
            if tokens:
                seen_token_sets.append(tokens)
        return deduped, seen_token_sets
    # Returns dedupe records used by final synthesis.
    def _dedupe_records(self, records):
        deduped = []; seen = set()
        for record in list(records or []):
            key = self._record_key(record)
            if not key[1] or key in seen:
                continue
            seen.add(key); deduped.append(record)
        return deduped
    # Returns whether record has concrete detail for final synthesis.
    def _record_has_concrete_detail(self, record) -> bool:
        return bool(CONCRETE_DETAIL_RE.search(str((record or {}).get("text") or "")))
    # Returns record concrete count for final synthesis.
    def _record_concrete_count(self, record) -> int:
        return len(CONCRETE_DETAIL_RE.findall(str((record or {}).get("text") or "")))
    # Returns chain records used by final synthesis.
    def _chain_records(self, chain):
        records = []
        if not chain or not isinstance(chain[0], dict):
            return records
        chain = self._truncate_structured_chain(chain)
        for step in chain[1:]:
            clause = self._step_clause(step)
            if clause:
                records.append({"text": clause, "source": str(step.get("source") or "").strip(), "role": "path",})
            for detail in list(step.get("specific_details", []) or []):
                if not isinstance(detail, dict):
                    continue
                text = self._clause_text(detail.get("text", ""))
                if not text:
                    continue
                records.append({"text": text, "source": str(detail.get("source") or "").strip(), "role": "specific_context",})
        return records
    # Formats argument for output.
    def _format_argument(self, chain) -> str:
        if isinstance(chain, str):
            return chain.strip()
        if not chain:
            return ""
        if isinstance(chain[0], dict):
            return self._record_chain_text(self._chain_records(chain))
        return self._sentence_chain_text(chain)
    # Returns source refs for final synthesis.
    def _source_refs(self, sources, source_map) -> str:
        return " ".join(f"[{source_map[display_source(source)]}]" for source in list(sources or []) if display_source(source) in source_map)
    # Builds record refs for final synthesis.
    def _record_refs(self, record, source_map) -> str:
        source = display_source((record or {}).get("source"))
        if not source or source not in source_map:
            return ""
        return f"[{source_map[source]}]"
    # Returns sentence chain text for final synthesis.
    def _sentence_chain_text(self, clauses, sources=None, source_map=None) -> str:
        source_refs = self._source_refs(sources, source_map or {}); sentences = []
        for clause in list(clauses or []):
            text = self._clause_text(clause)
            if not text:
                continue
            if source_refs and self._should_cite_clause(text):
                text = f"{text.rstrip('.')} {source_refs}."
            sentences.append(text)
        return " ".join(sentences)
    # Returns record chain text for final synthesis.
    def _record_chain_text(self, records, source_map=None, require_citation=False) -> str:
        sentences = []
        for record in list(records or []):
            text = self._clause_text((record or {}).get("text", ""))
            if not text:
                continue
            source = str((record or {}).get("source") or "").strip(); source_refs = self._source_refs([source], source_map or {})
            if require_citation and not source_refs:
                continue
            if source_refs:
                text = f"{text.rstrip('.')} {source_refs}."
            sentences.append(text)
        return " ".join(sentences)
    # Returns payload records used by final synthesis.
    def _payload_records(self, payload):
        records = []
        for record in list((payload or {}).get("clause_records", []) or []):
            if not isinstance(record, dict):
                continue
            text = self._clause_text(record.get("text", ""))
            if not text:
                continue
            if not is_usable_clause_text(text):
                continue
            records.append({"text": text, "source": str(record.get("source") or "").strip(), "role": str(record.get("role") or "path").strip() or "path", "specificity": list(record.get("specificity", []) or []),})
        return self._dedupe_records(records)
    # Returns payload role records used by final synthesis.
    def _payload_role_records(self, payload, role, require_citation=False):
        records = [record for record in self._payload_records(payload) if (record or {}).get("role") == role]
        if require_citation:
            records = [record for record in records if self._source_is_citable((record or {}).get("source"))]
        return records
    # Returns payload source count for final synthesis.
    def _payload_source_count(self, payload) -> int:
        sources = {display_source((record or {}).get("source")).rstrip("/").lower() for record in self._payload_records(payload) if self._source_is_citable((record or {}).get("source"))}
        return len([source for source in sources if source and source != "unknown"])
    # Returns payload concrete score for final synthesis.
    def _payload_concrete_score(self, payload) -> int:
        return sum(1 for record in self._payload_records(payload) if self._record_has_concrete_detail(record) and self._source_is_citable((record or {}).get("source")))
    # Returns payload path concrete score for final synthesis.
    def _payload_path_concrete_score(self, payload) -> int:
        return sum(1 for record in self._payload_role_records(payload, "path", require_citation=True) if self._record_has_concrete_detail(record))
    # Returns payload chain char count for final synthesis.
    def _payload_chain_char_count(self, payload) -> int:
        records = self._payload_role_records(payload, "path", require_citation=True)
        if not records:
            records = self._payload_role_records(payload, "path")
        return len(" ".join(str((record or {}).get("text") or "").strip() for record in records).strip())
    # Returns target side tokens for target matching.
    def _target_side_tokens(self, target_a="", target_b=""):
        return (set(target_tokens(target_a, min_length=2)), set(target_tokens(target_b, min_length=2)),)
    # Computes record target side counts for final synthesis.
    def _record_target_side_counts(self, record, target_a_tokens, target_b_tokens):
        record_tokens = self._record_tokens(record)
        return (len(record_tokens & set(target_a_tokens or [])), len(record_tokens & set(target_b_tokens or [])),)
    # Returns payload target bridge key for final synthesis.
    def _payload_target_bridge_key(self, payload, target_a="", target_b=""):
        target_a_tokens, target_b_tokens = self._target_side_tokens(target_a, target_b); path_records = self._payload_role_records(payload, "path", require_citation=True)
        if not path_records:
            path_records = self._payload_role_records(payload, "path")
        path_tokens = set(); both_side_records = 0; target_a_records = 0; target_b_records = 0
        for record in path_records:
            path_tokens.update(self._record_tokens(record)); a_hits, b_hits = self._record_target_side_counts(record, target_a_tokens, target_b_tokens)
            if a_hits:
                target_a_records += 1
            if b_hits:
                target_b_records += 1
            if a_hits and b_hits:
                both_side_records += 1
        covers_target_a = bool(path_tokens & target_a_tokens); covers_target_b = bool(path_tokens & target_b_tokens)
        return (both_side_records, covers_target_a and covers_target_b, target_b_records, target_a_records,)
    # Returns relationship record key for final synthesis.
    def _relationship_record_key(self, record, target_a_tokens, target_b_tokens):
        target_a_hits, target_b_hits = self._record_target_side_counts(record, target_a_tokens, target_b_tokens,)
        return (target_a_hits > 0 and target_b_hits > 0, target_b_hits > 0, target_a_hits > 0, self._record_has_concrete_detail(record), self._record_concrete_count(record), -len(self._record_tokens(record)),)
    # Returns payload rank key for final synthesis.
    def _payload_rank_key(self, payload, target_a="", target_b=""):
        return (*self._payload_target_bridge_key(payload, target_a, target_b), self._payload_path_concrete_score(payload), self._payload_concrete_score(payload), self._payload_source_count(payload), float((payload or {}).get("support_score", 0.0) or 0.0), self._payload_chain_char_count(payload),)
    # Returns payload mechanism tokens for final synthesis.
    def _payload_mechanism_tokens(self, payload, target_a="", target_b=""):
        target_token_set = set(target_tokens(f"{target_a} {target_b}", min_length=2)); path_text = " ".join(str((record or {}).get("text") or "") for record in self._payload_role_records(payload, "path")); tokens = set(target_tokens(path_text, min_length=2))
        return {token for token in tokens if token not in target_token_set and not token.isdigit()}
    # Computes token similarity for final synthesis.
    def _token_similarity(self, left, right):
        left = set(left or []); right = set(right or [])
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
    # Selects diverse payloads for final synthesis.
    def _select_diverse_payloads(self, payloads, target_a="", target_b="", limit=None):
        limit = max(1, int(limit or SYNTHESIS_ARGUMENT_LIMIT)); ranked = sorted(payloads, key=lambda payload: self._payload_rank_key(payload, target_a, target_b), reverse=True,); selected = []; selected_mechanisms = []; seen = set()
        # Builds a diversity signature for one candidate payload.
        def signature(payload):
            text = " ".join(str((record or {}).get("text") or "").strip().lower() for record in self._payload_role_records(payload, "path", require_citation=True)); route = str((payload or {}).get("route", "") or "").strip().lower(); endpoint = str((payload or {}).get("endpoint", "") or "").strip().lower()
            return route, endpoint, text
        # Attempts try append for final synthesis.
        def try_append(payload, enforce_diversity):
            key = signature(payload)
            if key in seen:
                return False
            mechanism_tokens = self._payload_mechanism_tokens(payload, target_a, target_b)
            if enforce_diversity and selected_mechanisms and mechanism_tokens:
                similarity = max(self._token_similarity(mechanism_tokens, selected_tokens) for selected_tokens in selected_mechanisms)
                if similarity >= PAYLOAD_DIVERSITY_THRESHOLD:
                    return False
            seen.add(key); selected.append(payload); selected_mechanisms.append(mechanism_tokens)
            return True
        for payload in ranked:
            try_append(payload, enforce_diversity=True)
            if len(selected) >= limit:
                return selected
        for payload in ranked:
            try_append(payload, enforce_diversity=False)
            if len(selected) >= limit:
                return selected
        return selected
    # Builds source map for final synthesis.
    def _build_source_map(self, payloads) -> dict:
        source_map = {}
        for payload in payloads:
            for record in self._payload_records(payload):
                if not self._source_is_citable((record or {}).get("source")):
                    continue
                source = display_source((record or {}).get("source"))
                if source and source != "unknown" and source not in source_map:
                    source_map[source] = len(source_map) + 1
        return source_map
    # Cleans strip existing source list for final synthesis.
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
    # Normalizes report heading for final synthesis.
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
            next_is_blank = index + 1 < len(lines) and not lines[index + 1].strip(); title = self._strip_terminal_source_refs(clean).rstrip(".").strip(); title_word_count = len(TEXT_TOKEN_RE.findall(title))
            if next_is_blank and 2 <= title_word_count <= 22:
                lines[index] = f"## {title}"
            break
        return "\n".join(lines).strip()
    # Cleans strip argument labels for final synthesis.
    def _strip_argument_labels(self, text):
        lines = []
        for line in str(text or "").splitlines():
            clean = line.strip()
            heading = re.match(r"^(#{1,4})\s+Argument\s+\d+\s*:\s*(.+)$", clean, flags=re.IGNORECASE)
            if heading:
                lines.append(f"{heading.group(1)} {heading.group(2).strip()}")
                continue
            lines.append(re.sub(r"\bArgument\s+\d+\s*:\s*", "", line, flags=re.IGNORECASE))
        return "\n".join(lines).strip()
    # Cleans strip template report headings for final synthesis.
    def _strip_template_report_headings(self, text):
        lines = []
        for line in str(text or "").splitlines():
            clean = line.strip()
            heading = re.match(r"^(#{1,4})\s+(.+)$", clean)
            if heading:
                heading_text = self._strip_terminal_source_refs(heading.group(2)).strip(" .:").lower()
                if heading_text in TEMPLATE_REPORT_HEADINGS:
                    continue
            elif clean:
                heading_text = self._strip_terminal_source_refs(clean).strip(" .:").lower()
                if heading_text in TEMPLATE_REPORT_HEADINGS:
                    continue
            lines.append(line)
        text = "\n".join(lines).strip()
        for pattern, replacement in PIPELINE_LANGUAGE_REPLACEMENTS:
            text = pattern.sub(replacement, text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    # Cleans strip structure language for final synthesis.
    def _strip_structure_language(self, text, target_a="", target_b=""):
        text = str(text or ""); target_labels = (("A", str(target_a or "").strip()), ("B", str(target_b or "").strip()))
        for label, target in target_labels:
            text = re.sub(
                rf"\bTarget\s+{label},\s*(?:the\s+)?([^,\n]{{2,180}}),\s*", r"\1 ", text, flags=re.IGNORECASE,
            )
            if target:
                text = re.sub(rf"\bTarget\s+{label}\b", target, text, flags=re.IGNORECASE)
            else:
                text = re.sub(rf"\bTarget\s+{label}\b", "", text, flags=re.IGNORECASE)
        if target_labels[0][1]:
            text = re.sub(r"\bthe\s+first\s+subject\b", target_labels[0][1], text, flags=re.IGNORECASE); text = re.sub(r"\bfirst\s+subject\b", target_labels[0][1], text, flags=re.IGNORECASE)
        if target_labels[1][1]:
            text = re.sub(r"\bthe\s+second\s+subject\b", target_labels[1][1], text, flags=re.IGNORECASE); text = re.sub(r"\bsecond\s+subject\b", target_labels[1][1], text, flags=re.IGNORECASE)
        text = re.sub(
            r"\b(?:the\s+)?(?:(?:available|provided|cited)\s+)?evidence\s+" r"(?:does|do)\s+not\s+(?:provide\s+)?(?:sufficient\s+)?support\s+for\s+the\s+claim\s+that\s+", "it is not established that ", text, flags=re.IGNORECASE,
        ); text = re.sub(
            r"\b(?:the\s+)?(?:(?:available|provided|cited)\s+)?evidence\s+" r"(?:does|do)\s+not\s+(?:establish|show|demonstrate|support)\s+(?:that\s+)?", "it is not established that ", text, flags=re.IGNORECASE,
        ); text = EVIDENCE_REPORTING_RE.sub("", text); text = FACT_REPORTING_RE.sub("", text); text = re.sub(
            r"\b(?:While|Although)\s+there\s+(?:are|is)\s+no\s+direct\s+causal\s+links?\s+between\s+[^,]+,\s*", "", text, flags=re.IGNORECASE,
        ); text = re.sub(
            r"(?m)(^|(?<=[.!?]\s))There\s+(?:are|is)\s+no\s+direct\s+causal\s+links?\s+between\s+[^.!?]+[.!?]\s*", "", text, flags=re.IGNORECASE,
        ); text = re.sub(r"\bno\s+direct\s+causal\s+links?\b", "an indirect relationship", text, flags=re.IGNORECASE); text = re.sub(r"\b(?:(?:available|provided|cited)\s+)?evidence\b", "cited details", text, flags=re.IGNORECASE); text = re.sub(r"\bthe cited details between\b", "the relationship between", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+([,.;:])", r"\1", text); text = re.sub(r"([.!?]\s+|\n)([a-z])", lambda match: f"{match.group(1)}{match.group(2).upper()}", text)
        text = re.sub(r"(?m)^(#{1,4}\s+)([a-z])", lambda match: f"{match.group(1)}{match.group(2).upper()}", text)
        if text[:1].islower():
            text = text[:1].upper() + text[1:]
        return re.sub(r"[ \t]{2,}", " ", text).strip()
    # Returns sentence claim tokens for final synthesis.
    def _sentence_claim_tokens(self, sentence):
        text = re.sub(r"\[\d+\]", "", str(sentence or ""))
        return set(target_tokens(text, min_length=3))
    # Splits report sentences into usable parts.
    def _split_report_sentences(self, paragraph):
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", str(paragraph or "").strip())
            if sentence.strip()
        ]
    # Cleans strip duplicate report sentences for final synthesis.
    def _strip_duplicate_report_sentences(self, text):
        blocks = re.split(r"\n{2,}", str(text or "").strip()); cleaned_blocks = []; seen_token_sets = []
        for block in blocks:
            clean = block.strip()
            if not clean:
                continue
            if clean.startswith("#"):
                cleaned_blocks.append(clean)
                continue
            kept_sentences = []
            for sentence in self._split_report_sentences(clean):
                tokens = self._sentence_claim_tokens(sentence)
                if self._claim_token_duplicate(tokens, seen_token_sets):
                    continue
                kept_sentences.append(sentence)
                if tokens:
                    seen_token_sets.append(tokens)
            if kept_sentences:
                cleaned_blocks.append(" ".join(kept_sentences))
        final_blocks = []
        for index, block in enumerate(cleaned_blocks):
            if block.startswith("#") and index > 0:
                has_body = any(
                    candidate and not candidate.startswith("#")
                    for candidate in cleaned_blocks[index + 1 :]
                ); next_heading_index = next(
                    (
                        offset
                        for offset, candidate in enumerate(cleaned_blocks[index + 1 :], start=index + 1)
                        if candidate.startswith("#")
                    ), None,
                )
                if next_heading_index is not None:
                    has_body = any(
                        candidate and not candidate.startswith("#")
                        for candidate in cleaned_blocks[index + 1 : next_heading_index]
                    )
                if not has_body:
                    continue
            final_blocks.append(block)
        return "\n\n".join(final_blocks).strip()
    # Returns whether sentence has source ref for final synthesis.
    def _sentence_has_source_ref(self, sentence):
        return bool(re.search(r"\[\d+\](?:\s*\[\d+\])*\s*[.!?]?$", sentence.strip()))
    # Cleans strip terminal source refs for final synthesis.
    def _strip_terminal_source_refs(self, sentence):
        sentence = sentence.strip()
        match = re.search(r"\s*(?:\[\d+\]\s*)+([.!?])?\s*$", sentence)
        if not match:
            return sentence
        punctuation = match.group(1) or ""; stripped = sentence[:match.start()].rstrip()
        if punctuation and stripped[-1:] not in ".!?":
            stripped = f"{stripped}{punctuation}"
        return stripped
    # Returns citation tokens for final synthesis.
    def _citation_tokens(self, text):
        return {token for token in TEXT_TOKEN_RE.findall(str(text or "").lower()) if len(token) > 2}
    # Returns best evidence refs for final synthesis.
    def _best_evidence_refs(self, sentence, records, source_map, limit=2):
        sentence_tokens = self._citation_tokens(sentence)
        if not sentence_tokens:
            return ""
        ranked = []; seen_refs = set()
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
            precision = len(overlap) / len(sentence_tokens); recall = len(overlap) / len(record_tokens); score = max(precision, 0.85 * recall)
            if score < 0.22:
                continue
            ranked.append((score, len(overlap), ref)); seen_refs.add(ref)
        ranked.sort(reverse=True)
        return " ".join(ref for _score, _overlap, ref in ranked[:limit])
    # Adds append ref to sentence for final synthesis.
    def _append_ref_to_sentence(self, sentence, ref):
        sentence = sentence.strip()
        if not sentence or self._sentence_has_source_ref(sentence):
            return sentence
        if sentence[-1:] in ".!?":
            return f"{sentence[:-1].rstrip()} {ref}{sentence[-1]}"
        return f"{sentence} {ref}."
    # Ensures ensure sentence citations for final synthesis.
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
            sentences = re.split(r"(?<=[.!?])\s+", paragraph); cited_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                base_sentence = self._strip_terminal_source_refs(sentence); refs = self._best_evidence_refs(base_sentence, records, source_map)
                if refs:
                    cited_sentences.append(self._append_ref_to_sentence(base_sentence, refs))
                    continue
                cited_sentences.append(base_sentence or sentence)
            paragraphs.append(" ".join(cited_sentences))
        return "\n\n".join(paragraphs).strip()
    # Adds append source list for final synthesis.
    def _append_source_list(self, text, source_map):
        if not source_map:
            return str(text or "").strip()
        source_lines = "\n".join(
            f"[{idx}]: {source}"
            for source, idx in sorted(source_map.items(), key=lambda item: item[1])
        )
        return f"{str(text or '').strip()}\n\n## Sources\n{source_lines}".strip()
    # Builds compact source map for final synthesis.
    def _compact_source_map(self, text, source_map):
        reverse_map = {idx: source for source, idx in source_map.items()}; old_to_new = {}; compact_source_map = {}
        for match in SOURCE_REF_RE.finditer(str(text or "")):
            old_idx = int(match.group(1)); source = reverse_map.get(old_idx)
            if not source or old_idx in old_to_new:
                continue
            new_idx = len(old_to_new) + 1; old_to_new[old_idx] = new_idx; compact_source_map[source] = new_idx
        if not old_to_new:
            return str(text or "").strip(), {}
        # Replaces replace ref for final synthesis.
        def replace_ref(match):
            old_idx = int(match.group(1)); new_idx = old_to_new.get(old_idx)
            return f"[{new_idx}]" if new_idx is not None else ""
        compact_text = SOURCE_REF_RE.sub(replace_ref, str(text or "")).strip(); compact_text = re.sub(r"\s+([.!?])", r"\1", compact_text); compact_text = re.sub(r"[ \t]{2,}", " ", compact_text)
        return compact_text, compact_source_map
    # Returns record tokens for final synthesis.
    def _record_tokens(self, record):
        return set(target_tokens(str((record or {}).get("text") or ""), min_length=2))
    # Returns context relevance key for final synthesis.
    def _context_relevance_key(self, record, path_token_set, target_a_tokens, target_b_tokens):
        record_tokens = self._record_tokens(record); path_overlap = len(record_tokens & set(path_token_set or [])); target_a_overlap = len(record_tokens & set(target_a_tokens or [])); target_b_overlap = len(record_tokens & set(target_b_tokens or [])); specificity = len(list((record or {}).get("specificity", []) or []))
        return (target_a_overlap > 0 and target_b_overlap > 0, target_b_overlap > 0, path_overlap > 0, path_overlap, target_a_overlap + target_b_overlap, self._record_has_concrete_detail(record), self._record_concrete_count(record), specificity, -len(record_tokens),)
    # Builds the final cited report from selected payloads and source records.
    def synthesize_relationship_report(self, target_a: str, target_b: str, payloads: list[dict]) -> str:
        payloads = [dict(item or {}) for item in list(payloads or []) if isinstance(item, dict)]
        payloads = [payload for payload in payloads if self._payload_role_records(payload, "path", require_citation=True) and (payload.get("endpoint_reached") or (len(self._payload_role_records(payload, "path", require_citation=True)) >= 2 and float(payload.get("match_target_a", 0.0) or 0.0) >= SYNTHESIS_TARGET_MATCH_THRESHOLD and float(payload.get("match_target_b", 0.0) or 0.0) >= SYNTHESIS_TARGET_MATCH_THRESHOLD))]
        if not payloads:
            return f"No successful cited relationship was found between {target_a or 'target A'} and {target_b or 'target B'}."
        relationship_payloads = [payload for payload in payloads if self._payload_target_bridge_key(payload, target_a, target_b)[1]]
        if relationship_payloads:
            payloads = relationship_payloads
        payloads = self._select_diverse_payloads(payloads, target_a=target_a, target_b=target_b, limit=SYNTHESIS_ARGUMENT_LIMIT,); source_map = self._build_source_map(payloads); note_blocks = []; evidence_records = []; used_claim_token_sets = []; target_a_tokens, target_b_tokens = self._target_side_tokens(target_a, target_b)
        for payload in payloads:
            records = self._payload_records(payload); evidence_records.extend(records); path_records = self._payload_role_records(payload, "path", require_citation=True,); path_records = sorted(path_records, key=lambda record: self._relationship_record_key(record, target_a_tokens, target_b_tokens,), reverse=True,)[:SYNTHESIS_MAX_CHAIN_CLAUSES]
            path_records, used_claim_token_sets = self._dedupe_similar_records(path_records, used_claim_token_sets,)
            if not path_records:
                continue
            path_keys = {self._record_key(record, include_role=False) for record in path_records}; context_pool = [record for record in self._payload_role_records(payload, "specific_context", require_citation=True) if self._record_key(record, include_role=False) not in path_keys]
            context_pool = [record for record in context_pool if self._record_target_side_counts(record, target_a_tokens, target_b_tokens)[1] > 0]; path_token_set = set()
            for record in path_records:
                path_token_set.update(self._record_tokens(record))
            context_records = sorted(context_pool, key=lambda record: self._context_relevance_key(record, path_token_set, target_a_tokens, target_b_tokens,), reverse=True,)[:SYNTHESIS_MAX_CHAIN_CLAUSES * 3]; context_records, used_claim_token_sets = self._dedupe_similar_records(context_records, used_claim_token_sets,)
            path_text = self._record_chain_text(path_records, source_map=source_map, require_citation=True); context_text = self._record_chain_text(context_records, source_map=source_map, require_citation=True)
            if not path_text:
                continue
            block_parts = [f"Core relationship facts: {path_text}"]
            if context_text:
                block_parts.append(f"Supporting factual context: {context_text}")
            note_blocks.append("\n".join(block_parts))
        digest = "\n\n".join(note_blocks)
        if not digest:
            return f"No citable relationship was found between {target_a or 'target A'} and {target_b or 'target B'}."
        source_lines = "\n".join(
            f"[{idx}]: {source}"
            for source, idx in sorted(source_map.items(), key=lambda item: item[1])
        )
        prompt = f"""
            Write a research report about how these subjects are connected:
            - "{target_a}"; - "{target_b}"

            Use only the cited facts below.

            Cited facts:
            {digest}

            Source index:
            {source_lines or "(no explicit sources available)"}

            Requirements:
            - Write a concise analytical report in formal prose, not bullets or a list of extracted facts.
            - Start with one neutral `##` title, then use 2 to 5 short descriptive body headings for the main themes in the report.
            - Make body headings content-specific and fact-derived; avoid generic labels like Introduction, Evidence, Concrete Details, Mechanism of Impact, Tradeoff, Time Period, Place, Metric, Policy Implications, Limitations, or Conclusion.; - Open with the strongest source-grounded thesis about how "{target_a}" and "{target_b}" are related.
            - Each body paragraph should make one substantive claim, cite the support for it, preserve concrete dates/numbers/names when available, and interpret how it connects "{target_a}" to "{target_b}".
            - Do not repeat the same mechanism, program, statistic, date, or source claim in multiple paragraphs; mention each distinct fact once and merge related support into the same paragraph.; - Treat core relationship facts as the main support and supporting factual context as extra detail.
            - Omit background about either subject alone unless the same paragraph explicitly ties it to the relationship between "{target_a}" and "{target_b}".; - Give more space to mechanisms, figures, dates, named places, policies, and time-bounded comparisons than to definitions; include at most one concise definition unless multiple definitions matter.
            - Preserve polarity: keep positive, negative, mixed, weak, indirect, or insufficient support distinct instead of blending opposite claims.; - Do not generalize beyond the provided place, period, metric, group, or wording; if two periods or metrics differ, say so.
            - If the facts describe the two subjects separately without bridging them, say the relationship is insufficiently supported.; - Cite each source-grounded sentence with the relevant markers already attached to the cited facts, like [1] or [1] [2], and do not cite headings.
            - Do not invent sources, mechanisms, causal claims, agreement levels, numbers, or hard facts.; - Use the actual subject names; never write labels such as Target A, Target B, first subject, second subject, the evidence, provided evidence, or available evidence.
            - Do not write phrases like "the evidence suggests" or "the facts show"; state the claim directly and cite it.; - Do not write stock disclaimers like "no direct causal link" or "no direct causal links"; when causality is weak, describe the specific indirect relationship the facts support.
            - Do not describe the method, input format, or selection process, and do not use pipeline words such as argument, chain, path, payload, record, evidence, extracted graph, sampled, selected, node, endpoint, or connector.; - End with a short measured conclusion matching the actual strength of the cited facts.

            Return only the finished report text.
        """
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": True, "options": {"temperature": 0.05, "top_p": 0.5,},}; response = requests.post(self.ollama_url, json=payload, stream=True, timeout=(self.connect_timeout, self.read_timeout),); response.raise_for_status(); chunks = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line); chunk = data.get("response", "")
                if chunk:
                    chunks.append(chunk)
                if data.get("done"):
                    break
            final_text = "".join(chunks).strip()
            if not final_text:
                return "Error: No response from model."
            final_text = self._strip_existing_source_list(final_text); final_text = self._normalize_report_heading(final_text); final_text = self._strip_argument_labels(final_text); final_text = self._strip_template_report_headings(final_text); final_text = self._strip_structure_language(final_text, target_a, target_b)
            final_text = self._ensure_sentence_citations(final_text, source_map, evidence_records); final_text = self._strip_structure_language(final_text, target_a, target_b); final_text = self._strip_duplicate_report_sentences(final_text); final_text, used_source_map = self._compact_source_map(final_text, source_map)
            return self._append_source_list(final_text, used_source_map)
        except requests.exceptions.ConnectionError:
            return f"Synthesis Failed: Could not connect to Ollama at {self.ollama_url}. Start Ollama and retry."
        except requests.exceptions.ReadTimeout:
            return f"Synthesis Failed: Ollama did not finish within {int(self.read_timeout)} seconds. Increase OLLAMA_READ_TIMEOUT or use a smaller/faster model."
        except Exception as e:
            return f"Synthesis Failed: {str(e)}"
