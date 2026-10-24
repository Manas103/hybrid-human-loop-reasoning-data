import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_problems import generate_reasoning_problems, make_cot_prompt


def test_deterministic():
    a = generate_reasoning_problems(50, seed=9)
    b = generate_reasoning_problems(50, seed=9)
    assert [p.final_true for p in a] == [p.final_true for p in b]


def test_step1_consistent_with_final():
    for p in generate_reasoning_problems(100, seed=1):
        assert p.step1_true > 0
        assert p.final_true >= 0
        assert p.final_true <= p.step1_true


def test_prompt_requests_exact_three_line_format():
    p = generate_reasoning_problems(1, seed=2)[0]
    prompt = make_cot_prompt(p)
    assert "Step 1 result:" in prompt
    assert "Step 2 result:" in prompt
    assert "Answer:" in prompt
