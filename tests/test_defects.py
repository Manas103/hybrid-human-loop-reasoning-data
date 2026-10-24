import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defects import check_defects, parse_trace


def test_clean_trace_is_not_defective():
    trace = "Step 1 result: 20\nStep 2 result: 15\nAnswer: 15"
    r = check_defects(trace, step1_true=20, final_true=15)
    assert r.wrong_answer is False
    assert r.unfaithful_step is False
    assert r.is_defective is False


def test_wrong_answer_is_defective():
    trace = "Step 1 result: 20\nStep 2 result: 14\nAnswer: 14"
    r = check_defects(trace, step1_true=20, final_true=15)
    assert r.wrong_answer is True
    assert r.is_defective is True


def test_unfaithful_step_with_correct_final_answer_is_still_defective():
    """Right answer, wrong path: compensating errors, caught only by step1_true."""
    trace = "Step 1 result: 19\nStep 2 result: 15\nAnswer: 15"
    r = check_defects(trace, step1_true=20, final_true=15)
    assert r.wrong_answer is False
    assert r.unfaithful_step is True
    assert r.is_defective is True


def test_unparseable_trace_counts_as_wrong_answer():
    r = check_defects("I am not sure.", step1_true=20, final_true=15)
    assert r.wrong_answer is True
    assert r.is_defective is True


def test_parse_trace_extracts_both_numbers():
    step1, answer = parse_trace("Step 1 result: 7\nStep 2 result: 3\nAnswer: 3")
    assert step1 == 7
    assert answer == 3
