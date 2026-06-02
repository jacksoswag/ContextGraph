from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eval.assertions import CheckResult, answer_text, case_passed, output_keys, summarize_support, summarize_top_active
from eval.cases import EvalCase, fact_key
from testing_manager import ResultRecord, write_latest_result

KEY_REPORT_LIMIT = 16


@dataclass(frozen=True)
class EvalRow:
    case: EvalCase
    result: dict[str, object] | None
    checks: tuple[CheckResult, ...]
    blocked: str = ""

    @property
    def passed(self) -> bool:
        return not self.blocked and case_passed(self.checks)


def write_eval_report(
    test_type: str,
    test_name: str,
    rows: tuple[EvalRow, ...],
    *,
    started_at: datetime,
    finished_at: datetime,
    thresholds: dict[str, float] | None = None,
    extra_lines: tuple[str, ...] = (),
) -> Path:
    status = suite_status(rows, thresholds=thresholds)
    body = report_body(rows, thresholds=thresholds, extra_lines=extra_lines)
    return write_latest_result(
        test_type,
        ResultRecord(
            test_type=test_name,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            body=body,
        ),
    )


def suite_status(rows: tuple[EvalRow, ...], *, thresholds: dict[str, float] | None = None) -> str:
    if not rows or any(row.blocked for row in rows):
        return "FAIL"
    thresholds = thresholds or {}
    grouped: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        grouped[row.case.category].append(row)
    for category, minimum in thresholds.items():
        category_rows = grouped.get(category, [])
        if not category_rows:
            return "FAIL"
        if _rate(category_rows) < minimum:
            return "FAIL"
    for category, category_rows in grouped.items():
        if category not in thresholds and not all(row.passed for row in category_rows):
            return "FAIL"
    return "PASS"


def report_body(
    rows: tuple[EvalRow, ...],
    *,
    thresholds: dict[str, float] | None = None,
    extra_lines: tuple[str, ...] = (),
) -> str:
    thresholds = thresholds or {}
    counts = Counter(row.case.category for row in rows)
    passed_counts = Counter(row.case.category for row in rows if row.passed)
    total_passed = sum(1 for row in rows if row.passed)
    lines = [
        f"Strict pass count: {total_passed}/{len(rows)}",
        "PASS/FAIL by category:",
    ]
    for category in sorted(counts):
        rate = passed_counts[category] / counts[category] if counts[category] else 0.0
        threshold_text = f" threshold={thresholds[category]:.0%}" if category in thresholds else ""
        lines.append(f"  {category:<18} {passed_counts[category]:>3}/{counts[category]:<3} {rate:>6.1%}{threshold_text}")
    if extra_lines:
        lines.extend(("", *extra_lines))
    lines.extend(("", "Failures:", ""))
    failure_rows = [row for row in rows if not row.passed]
    if not failure_rows:
        lines.append("  none")
    for row in failure_rows:
        lines.extend(_failure_lines(row))
        lines.append("")
    lines.extend(("", "Detail:", ""))
    for row in rows:
        label = "PASS" if row.passed else "FAIL"
        lines.append(f"[{label}] {row.case.category}::{row.case.id}")
        lines.append(f"  prompt: {row.case.prompt!r}")
        lines.append(f"  answer: {answer_text(row.result or {})!r}")
    return "\n".join(lines).rstrip()


def _failure_lines(row: EvalRow) -> list[str]:
    case = row.case
    result = row.result or {}
    failed = [check for check in row.checks if not check.passed]
    reason = row.blocked or "; ".join(f"{check.name}: {check.reason}" for check in failed)
    expected_keys = tuple(fact_key(fact) for fact in case.expected_facts)
    forbidden_keys = tuple(fact_key(fact) for fact in case.forbidden_facts)
    actual_keys = tuple(sorted(output_keys(result)))
    forbidden_collisions = tuple(key for key in forbidden_keys if key in actual_keys)
    return [
        f"[FAIL] {case.category}::{case.id}",
        f"  failure_reason: {reason}",
        f"  prompt: {case.prompt!r}",
        f"  expected output keys: {expected_keys}",
        f"  actual output keys:   {_key_summary(actual_keys)}",
        f"  required terms:       {case.expected_targets}",
        f"  forbidden terms hit:  {forbidden_collisions}",
        f"  actual answer:        {answer_text(result)!r}",
        f"  top support edges:    {summarize_support(result)}",
        f"  top active values:    {summarize_top_active(result)}",
    ]


def _rate(rows: list[EvalRow]) -> float:
    return sum(1 for row in rows if row.passed) / len(rows) if rows else 0.0


# Keep full-memory failures readable without hiding the output fanout size.
def _key_summary(keys: tuple[str, ...], *, limit: int = KEY_REPORT_LIMIT) -> str:
    if len(keys) <= limit:
        return repr(keys)
    head = keys[:limit]
    return f"{head!r} ... +{len(keys) - limit} more"
