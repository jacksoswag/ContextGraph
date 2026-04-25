from datetime import datetime
from pathlib import Path

from constants import (
    FINAL_ARGUMENT_LIMIT,
    PHASE_IDLE,
    REPORT_SHM_BYTES,
    RESULTS_DIR,
    THOUGHT_FALLBACK_COMPLETED_LIMIT,
)


def _unique_results_dir(root=RESULTS_DIR):
    base = Path(root)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = base / stamp
    suffix = 2
    while path.exists():
        path = base / f"{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def _read_text_file(path):
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_report_to_shm(self, report):
    report_bytes = str(report or "").encode("utf-8")[: (REPORT_SHM_BYTES - 1)]
    self.shm_report.buf[:len(report_bytes)] = report_bytes
    self.shm_report.buf[len(report_bytes):len(report_bytes) + 1] = b"\x00"
    return len(report_bytes)


def export_final_results(self, synthesis_text):
    self.flush_conn_log()
    output_dir = _unique_results_dir()
    (output_dir / "synthesis.txt").write_text(str(synthesis_text or ""), encoding="utf-8")
    (output_dir / "connections.txt").write_text(_read_text_file("connection_report.txt"), encoding="utf-8")
    (output_dir / "arguments.txt").write_text(_read_text_file("argument_report.txt"), encoding="utf-8")
    return output_dir

def write_argument_report(self, active_histories, sampled_histories, is_final=False, successful_payloads=None):
    with open("argument_report.txt", "w") as f:
        target_a = self.current_target_a or "(none)"
        target_b = self.current_target_b or "(none)"
        combined_target = self.current_target or "(none)"
        target_a_focus = ", ".join(self.target_a_focus_phrases) or "(none)"
        target_b_focus = ", ".join(self.target_b_focus_phrases) or "(none)"
        tense_preference = getattr(self, "current_tense_preference", "none") or "none"
        phase = "final" if is_final else "interim"

        f.write("--- ARGUMENT REPORT ---\n")
        f.write(f"Target A is {target_a}.\n")
        f.write(f"Target B is {target_b}.\n")
        f.write(f"{target_a} and {target_b} form the combined target {combined_target}.\n")
        f.write(f"{target_a} has {len(self.target_a_queries)} target queries.\n")
        f.write(f"{target_b} has {len(self.target_b_queries)} target queries.\n")
        f.write(f"The bridge between them has {len(self.bridge_queries)} bridge queries.\n")
        f.write(f"{target_a} is focused through {target_a_focus}.\n")
        f.write(f"{target_b} is focused through {target_b_focus}.\n")
        f.write(f"The thought process favors {tense_preference} tense.\n")
        f.write(f"The report phase is {phase}.\n")
        f.write(f"The active thought set has {len(active_histories)} histories.\n")
        f.write(f"The sampled thought set has {len(sampled_histories)} histories.\n\n")

        successful_payloads = list(successful_payloads or [])
        if successful_payloads:
            f.write("--- SUCCESSFUL RELATIONSHIPS ---\n")
            for idx, payload in enumerate(successful_payloads, 1):
                f.write(f"[SUCCESS {idx}] {payload}\n")
            f.write("\n")

        if not active_histories:
            f.write("No active thought histories were available yet.\n")
            return

        for i, chain in enumerate(sampled_histories, 1):
            formatted = self.synthesizer._format_argument(chain)
            f.write(f"[ARG {i}] {formatted or '(empty chain)'}\n")
            f.write(f"[RAW {i}] {chain}\n\n")


def run_final_synthesis(self):
    if not self.current_target or not self.current_command_id:
        return
    if self.current_command_id in self.finalized_command_ids:
        return

    report = ""
    self.stop_thought_workers()
    print("[SYNTHESIS] Swarm stabilized. Running final synthesis...")
    stats = self.thought_swarm_stats()
    print(
        f"[SYNTHESIS] Thought stats before fallback: "
        f"{stats['successful']} successful / {stats['completed']} completed / {stats['alive']} alive / "
        f"{stats['total']} total across {stats['seeds']} seeds."
    )
    successful_thoughts = self.successful_thoughts()
    _completed_thoughts, all_histories = self.completed_thought_histories()

    if not successful_thoughts and stats["completed"] < THOUGHT_FALLBACK_COMPLETED_LIMIT:
        print("[SYNTHESIS] No successful chains yet. Running synchronous thought bootstrap...")
        _completed_thoughts, all_histories = self.bootstrap_thought_histories()
        successful_thoughts = self.successful_thoughts()
        stats = self.thought_swarm_stats()
        print(
            f"[SYNTHESIS] Thought stats after fallback: "
            f"{stats['successful']} successful / {stats['completed']} completed / {stats['alive']} alive / "
            f"{stats['total']} total across {stats['seeds']} seeds."
        )
    elif not successful_thoughts:
        print(
            "[SYNTHESIS] No successful chains after threaded search. "
            "Skipping synchronous bootstrap to avoid repeating a large failed traversal."
        )

    if successful_thoughts:
        ranked_successful = sorted(
            successful_thoughts,
            key=lambda thought: (
                float((getattr(thought, "success_payload", {}) or {}).get("support_score", 0.0)),
                len(getattr(thought, "collected_sources", [])),
                len(getattr(thought, "collected_notes", [])),
            ),
            reverse=True,
        )
        selected_successful = ranked_successful[:FINAL_ARGUMENT_LIMIT]
        histories = [thought.history for thought in selected_successful]
        payloads = [thought.success_payload for thought in selected_successful if thought.success_payload]
        print(
            f"[SYNTHESIS] Running for '{self.current_target}' with "
            f"{len(successful_thoughts)} successful chains..."
        )
        self.write_argument_report(
            all_histories,
            histories,
            is_final=True,
            successful_payloads=payloads,
        )

        report = self.synthesizer.synthesize_relationship_report(
            self.current_target_a,
            self.current_target_b,
            payloads,
        )
        report_length = _write_report_to_shm(self, report)
        print(f"[SYNTHESIS] Report updated ({report_length} bytes).")
    else:
        self.write_argument_report(all_histories, [], is_final=True)
        report = (
            f"No successful relationship chains were found between "
            f"{self.current_target_a or 'target A'} and {self.current_target_b or 'target B'}."
        )
        _write_report_to_shm(self, report)
        print("[SYNTHESIS] No successful thought chains. Returning to idle.")

    try:
        output_dir = export_final_results(self, report)
        print(f"[SYNTHESIS] Exported results to {output_dir}")
    except Exception as exc:
        print(f"[SYNTHESIS] Results export failed: {exc}")

    self.finalized_command_ids.add(self.current_command_id)
    self.shm_status.buf[0] = PHASE_IDLE
