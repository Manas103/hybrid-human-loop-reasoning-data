"""Generates N chain-of-thought traces, scores each for defects and
confidence, and stores everything in DuckDB.

Chunked and resumable: each invocation generates at most --chunk-size new
traces and appends to a jsonl sidecar, so a crash loses at most one chunk.
Rerun the same command until it reports completion.

Usage:
  python generate_traces.py --n 1500 --seed 30 --out-db data/traces.duckdb --chunk-size 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defects import check_defects
from model_io import generate_with_confidence, load_base_model, load_tokenizer, render_prompt
from reasoning_problems import generate_reasoning_problems, make_cot_prompt
from store import connect, count_traces, insert_traces

MAX_NEW_TOKENS = 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--out-db", default="data/traces.duckdb")
    ap.add_argument("--jsonl", default="data/traces.items.jsonl")
    ap.add_argument("--chunk-size", type=int, default=24)
    args = ap.parse_args()

    problems = generate_reasoning_problems(args.n, seed=args.seed)

    jsonl_path = Path(args.jsonl)
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
        model = load_base_model()
        rendered = [render_prompt(tok, make_cot_prompt(p)) for p in to_run]

        t0 = time.time()
        results = generate_with_confidence(model, tok, rendered, max_new_tokens=MAX_NEW_TOKENS)
        elapsed = time.time() - t0

        with jsonl_path.open("a") as f:
            for problem, (completion, confidence) in zip(to_run, results):
                d = check_defects(completion, problem.step1_true, problem.final_true)
                record = {
                    "id": problem.id,
                    "prompt": problem.prompt,
                    "noun": problem.noun,
                    "step1_true": problem.step1_true,
                    "final_true": problem.final_true,
                    "completion": completion,
                    "confidence": confidence,
                    "wrong_answer": d.wrong_answer,
                    "unfaithful_step": d.unfaithful_step,
                    "is_defective": d.is_defective,
                }
                f.write(json.dumps(record) + "\n")
        print(f"chunk done: {len(to_run)} traces in {elapsed:.1f}s "
              f"({len(done_ids) + len(to_run)}/{len(problems)} total)")

    still_remaining = len(problems) - (len(done_ids) + len(to_run))
    if still_remaining > 0:
        print(f"REMAINING {still_remaining} items; rerun this command to continue.")
        return

    con = connect(args.out_db)
    records = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    insert_traces(con, records)
    print(f"inserted {count_traces(con)} traces into {args.out_db}")


if __name__ == "__main__":
    main()
