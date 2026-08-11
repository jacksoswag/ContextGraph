from datetime import datetime; from pathlib import Path; import re; from engine.common.constants import (PHASE_IDLE, REPORT_SHM_BYTES, RESULTS_DIR, SYNTHESIS_ARGUMENT_LIMIT,); from engine.extract.word_info_map import literal_from_index; from engine.extract.target_text import target_tokens; MECHANISM_DIVERSITY_THRESHOLD = 0.72; MECHANISM_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Returns unique results dir for reporting.
def _unique_results_dir(root=RESULTS_DIR):
    base = Path(root); stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); path = base / stamp; suffix = 2
    while path.exists():
        path = base / f"{stamp}_{suffix}"; suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path
# Reads text file from storage or shared memory.
def _read_text_file(path):
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")
# Writes report to shm to storage or shared memory.
def _write_report_to_shm(self, report):
    report_bytes = str(report or "").encode("utf-8")[: (REPORT_SHM_BYTES - 1)]; self.shm_report.buf[:len(report_bytes)] = report_bytes; self.shm_report.buf[len(report_bytes):len(report_bytes) + 1] = b"\x00"
    return len(report_bytes)
# Returns thought chain char count for reporting.
def _thought_chain_char_count(self, thought):
    payload = getattr(thought, "success_payload", None)
    if isinstance(payload, dict):
        return self.synthesizer._payload_chain_char_count(payload)
    return len(self.synthesizer._format_argument(getattr(thought, "history", [])))
# Returns selected thought source count for reporting.
def _selected_thought_source_count(self, thought):
    payload = getattr(thought, "success_payload", None)
    if isinstance(payload, dict):
        return self.synthesizer._payload_source_count(payload)
    return len(getattr(thought, "collected_sources", []))
# Returns selected thought concrete count for reporting.
def _selected_thought_concrete_count(self, thought):
    payload = getattr(thought, "success_payload", None)
    if isinstance(payload, dict):
        return self.synthesizer._payload_concrete_score(payload)
    return len(getattr(thought, "collected_notes", []))
# Returns selected thought path concrete count for reporting.
def _selected_thought_path_concrete_count(self, thought):
    payload = getattr(thought, "success_payload", None)
    if isinstance(payload, dict):
        return self.synthesizer._payload_path_concrete_score(payload)
    count = 0
    for step in list(getattr(thought, "history", []) or [])[1:]:
        if not isinstance(step, dict):
            continue
        fields = (step.get("subject_specifics", []), step.get("predicate_specifics", []), step.get("connection_specifics", []),)
        if any(list(field or []) for field in fields):
            count += 1
    return count
# Returns thought rank key for reporting.
def _thought_rank_key(self, thought):
    payload = getattr(thought, "success_payload", None); target_bridge_key = (self.synthesizer._payload_target_bridge_key(payload, getattr(self, "current_target_a", ""), getattr(self, "current_target_b", ""),) if isinstance(payload, dict) else (0, False, 0, 0))
    return (*target_bridge_key, _selected_thought_path_concrete_count(self, thought), _selected_thought_concrete_count(self, thought), _selected_thought_source_count(self, thought), float((payload or {}).get("support_score", 0.0)), _thought_chain_char_count(self, thought), len(getattr(thought, "collected_notes", [])),)
# Returns thought target tokens for reporting.
def _thought_target_tokens(self):
    target_text = f"{getattr(self, 'current_target_a', '')} {getattr(self, 'current_target_b', '')}"
    return set(target_tokens(target_text, min_length=2))
# Returns thought mechanism tokens for reporting.
def _thought_mechanism_tokens(self, thought):
    target_token_set = _thought_target_tokens(self); texts = []; history = list(getattr(thought, "history", []) or [])
    if history and isinstance(history[0], dict):
        for step in history[1:]:
            if not isinstance(step, dict):
                continue
            try:
                relation_id = int(step.get("relation_id", -1) or -1)
            except (TypeError, ValueError):
                relation_id = -1
            relation = literal_from_index(relation_id) or ""; texts.extend([step.get("subject", ""), relation, step.get("predicate", ""),])
    payload = getattr(thought, "success_payload", None)
    if isinstance(payload, dict):
        texts.extend(str((record or {}).get("text") or "") for record in self.synthesizer._payload_role_records(payload, "path"))
    tokens = set(target_tokens(" ".join(str(text or "") for text in texts), min_length=2))
    return {token for token in tokens if token not in target_token_set and not token.isdigit() and len(MECHANISM_TOKEN_RE.findall(token)) > 0}
# Computes mechanism similarity for reporting.
def _mechanism_similarity(left, right):
    left = set(left or []); right = set(right or [])
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
# Builds thought payload signature for reporting.
def _thought_payload_signature(self, thought):
    payload = getattr(thought, "success_payload", None)
    if isinstance(payload, dict):
        records = self.synthesizer._payload_role_records(payload, "path"); text = " ".join(str((record or {}).get("text") or "").strip().lower() for record in records).strip(); endpoint = str(payload.get("endpoint", "") or "").strip().lower(); route = str(payload.get("route", "") or "").strip().lower()
        if text:
            return (route, endpoint, text)
    return tuple(str(step or "").strip().lower() for step in getattr(thought, "history", []) or [])
# Selects successful thoughts for reporting.
def _select_successful_thoughts(self, thoughts):
    endpoint_thoughts = [thought for thought in thoughts if getattr(thought, "termination_reason", "") == "endpoint"]; ranked = sorted(endpoint_thoughts, key=lambda thought: _thought_rank_key(self, thought), reverse=True); selected = []; selected_mechanisms = []; seen = set(); limit = max(1, int(SYNTHESIS_ARGUMENT_LIMIT))
    # Attempts try append for reporting.
    def try_append(thought, enforce_diversity):
        signature = _thought_payload_signature(self, thought)
        if signature in seen:
            return False
        mechanism_tokens = _thought_mechanism_tokens(self, thought)
        if enforce_diversity and selected_mechanisms and mechanism_tokens:
            similarity = max(_mechanism_similarity(mechanism_tokens, selected_tokens) for selected_tokens in selected_mechanisms)
            if similarity >= MECHANISM_DIVERSITY_THRESHOLD:
                return False
        seen.add(signature); selected.append(thought); selected_mechanisms.append(mechanism_tokens)
        return True
    for thought in ranked:
        try_append(thought, enforce_diversity=True)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for thought in ranked:
            try_append(thought, enforce_diversity=False)
            if len(selected) >= limit:
                break
    return selected
# Persists final arguments and synthesis into a timestamped results folder.
def export_final_results(self, synthesis_text):
    self.flush_conn_log(); output_dir = _unique_results_dir(); (output_dir / "synthesis.txt").write_text(str(synthesis_text or ""), encoding="utf-8"); (output_dir / "connections.txt").write_text(_read_text_file("connection_report.txt"), encoding="utf-8"); (output_dir / "arguments.txt").write_text(_read_text_file("argument_report.txt"), encoding="utf-8")
    return output_dir
# Writes live thought diagnostics to the argument report file for debugging and synthesis.
def write_argument_report(self, active_histories, sampled_histories, is_final=False, successful_payloads=None):
    with open("argument_report.txt", "w") as f:
        target_a = self.current_target_a or "(none)"; target_b = self.current_target_b or "(none)"; combined_target = self.current_target or "(none)"; target_a_focus = ", ".join(self.target_a_focus_phrases) or "(none)"; target_b_focus = ", ".join(self.target_b_focus_phrases) or "(none)"; phase = "final" if is_final else "interim"
        f.write("--- ARGUMENT REPORT ---\n"); f.write(f"Target A is {target_a}.\n"); f.write(f"Target B is {target_b}.\n"); f.write(f"{target_a} and {target_b} form the combined target {combined_target}.\n"); f.write(f"{target_a} has {len(self.target_a_queries)} target queries.\n"); f.write(f"{target_b} has {len(self.target_b_queries)} target queries.\n")
        f.write(f"The bridge between them has {len(self.bridge_queries)} bridge queries.\n"); f.write(f"{target_a} query expansion focus phrases are {target_a_focus}.\n"); f.write(f"{target_b} query expansion focus phrases are {target_b_focus}.\n"); f.write(f"The report phase is {phase}.\n")
        f.write(f"The active thought set has {len(active_histories)} histories.\n"); f.write(f"The sampled thought set has {len(sampled_histories)} histories.\n\n"); successful_payloads = list(successful_payloads or [])
        if successful_payloads:
            f.write("--- SUCCESSFUL RELATIONSHIPS ---\n")
            for idx, payload in enumerate(successful_payloads, 1):
                f.write(f"[SUCCESS {idx}] {payload}\n")
            f.write("\n")
        if not active_histories:
            f.write("No active thought histories were available yet.\n")
            return
        for i, chain in enumerate(sampled_histories, 1):
            formatted = self.synthesizer._format_argument(chain); f.write(f"[ARG {i}] {formatted or '(empty chain)'}\n"); f.write(f"[RAW {i}] {chain}\n\n")
# Selects successful thoughts, runs synthesis, writes results, and updates shared report memory.
def run_final_synthesis(self):
    if not self.current_target or not self.current_command_id:
        return
    if self.current_command_id in self.finalized_command_ids:
        return
    self.stop_thought_workers(); print("[SYNTHESIS] Swarm stabilized. Running final synthesis..."); stats = self.thought_swarm_stats(); print(f"[SYNTHESIS] Thought stats: " f"{stats['successful']} successful / {stats['completed']} completed / {stats['alive']} alive / " f"{stats['total']} total across {stats['seeds']} seeds.")
    successful_thoughts = self.successful_thoughts(); all_histories = self.completed_thought_histories()[1]
    if successful_thoughts:
        selected_successful = _select_successful_thoughts(self, successful_thoughts); histories = [thought.history for thought in selected_successful]; payloads = [thought.success_payload for thought in selected_successful if thought.success_payload]; selected_chars = sum(_thought_chain_char_count(self, thought) for thought in selected_successful)
        print(f"[SYNTHESIS] Running for '{self.current_target}' with " f"the merged top {len(selected_successful)} of {len(successful_thoughts)} endpoint chains " f"({selected_chars} distinct path characters)..."); self.write_argument_report(all_histories, histories, is_final=True, successful_payloads=payloads,)
        report = self.synthesizer.synthesize_relationship_report(self.current_target_a, self.current_target_b, payloads,); report_length = _write_report_to_shm(self, report); print(f"[SYNTHESIS] Report updated ({report_length} bytes).")
    else:
        self.write_argument_report(all_histories, [], is_final=True); report = (f"No successful relationship chains were found between " f"{self.current_target_a or 'target A'} and {self.current_target_b or 'target B'}."); _write_report_to_shm(self, report); print("[SYNTHESIS] No successful thought chains. Returning to idle.")
    try:
        output_dir = export_final_results(self, report); print(f"[SYNTHESIS] Exported results to {output_dir}")
    except Exception as exc:
        print(f"[SYNTHESIS] Results export failed: {exc}")
    self.finalized_command_ids.add(self.current_command_id); self.shm_status.buf[0] = PHASE_IDLE
