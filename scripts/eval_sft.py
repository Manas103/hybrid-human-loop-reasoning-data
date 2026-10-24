"""Evaluates a base or SFT-adapted model's task accuracy on a held-out test
set of reasoning problems, generated with a seed disjoint from training.

Usage:
  python eval_sft.py --adapter runs/sft_budget/adapter --out docs/eval_sft_budget.json
  python eval_sft.py --adapter none --out docs/eval_sft_base.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defects import check_defects
from model_io import generate_with_confidence, load_base_model, load_lora_model, load_tokenizer, render_prompt
from reasoning_problems import generate_reasoning_problems, make_cot_prompt

TEST_SEED = 999999  # disjoint from any generation seed used to build training traces
MAX_NEW_TOKENS = 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-size", type=int, default=24)
    args = ap.parse_args()

    problems = generate_reasoning_problems(args.n, seed=TEST_SEED)

    jsonl_path = Path(args.out).with_suffix(".items.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass  # a crash mid-write can truncate the last line; drop it and regenerate that item

    remaining = [p for p in problems if p.id not in done_ids]
    to_run = remaining[: args.chunk_size]

    if to_run:
        tok = load_tokenizer()
        model = load_base_model() if args.adapter == "none" else load_lora_model(args.adapter)
        rendered = [render_prompt(tok, make_cot_prompt(p)) for p in to_run]

        t0 = time.time()
        results = generate_with_confidence(model, tok, rendered, max_new_tokens=MAX_NEW_TOKENS)
        elapsed = time.time() - t0

        with jsonl_path.open("a") as f:
            for problem, (completion, confidence) in zip(to_run, results):
                d = check_defects(completion, problem.step1_true, problem.final_true)
                record = {
                    "id": problem.id,
                    "completion": completion,
                    "confidence": confidence,
                    "wrong_answer": d.wrong_answer,
                    "unfaithful_step": d.unfaithful_step,
                    "is_defective": d.is_defective,
                }
                f.write(json.dumps(record) + "\n")
        print(f"chunk done: {len(to_run)} items in {elapsed:.1f}s "
              f"({len(done_ids) + len(to_run)}/{len(problems)} total)")

    still_remaining = len(problems) - (len(done_ids) + len(to_run))
    if still_remaining > 0:
        print(f"REMAINING {still_remaining} items; rerun this command to continue.")
        return

    all_records = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        all_records[rec["id"]] = rec
    records = [all_records[p.id] for p in problems]
    n = len(records)
    accuracy_pct = 100.0 * sum(1 for r in records if not r["wrong_answer"]) / n
    faithful_pct = 100.0 * sum(1 for r in records if not r["unfaithful_step"]) / n

    summary = {
        "adapter": args.adapter,
        "n": n,
        "task_accuracy_pct": accuracy_pct,
        "faithful_step_pct": faithful_pct,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
