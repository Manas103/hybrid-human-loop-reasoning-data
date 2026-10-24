"""Detects two independent kinds of trace defect from a generated CoT trace.

wrong_answer: the final number does not match the generator's true answer.
unfaithful_step: the stated step-1 result does not match the generator's
true intermediate value, even if the final answer happens to be right. A
trace can be unfaithful and still correct by coincidence (compensating
errors); this is exactly the case a final-answer-only check would miss.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STEP1_RE = re.compile(r"Step 1 result:\s*(-?\d+)", re.IGNORECASE)
STEP2_RE = re.compile(r"Step 2 result:\s*(-?\d+)", re.IGNORECASE)
ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)", re.IGNORECASE)


@dataclass
class DefectReport:
    parsed_step1: int | None
    parsed_answer: int | None
    wrong_answer: bool
    unfaithful_step: bool
    is_defective: bool


def parse_trace(trace_text: str) -> tuple[int | None, int | None]:
    m1 = STEP1_RE.search(trace_text)
    m_ans = ANSWER_RE.search(trace_text)
    step1 = int(m1.group(1)) if m1 else None
    answer = int(m_ans.group(1)) if m_ans else None
    return step1, answer


def check_defects(trace_text: str, step1_true: int, final_true: int) -> DefectReport:
    step1, answer = parse_trace(trace_text)
    wrong_answer = answer != final_true
    unfaithful_step = step1 != step1_true
    return DefectReport(
        parsed_step1=step1,
        parsed_answer=answer,
        wrong_answer=wrong_answer,
        unfaithful_step=unfaithful_step,
        is_defective=wrong_answer or unfaithful_step,
    )
