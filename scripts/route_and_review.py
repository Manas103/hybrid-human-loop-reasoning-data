"""Routes the worst-confidence slice of traces to simulated review, then
builds three training sets: unfiltered (no review), budget (review only the
routed slice), and ceiling (review everything, same simulated annotator).

Also runs a 300-trace calibration check: review_sim applied to a held-out
slice, measured against its own true labels, to report the realized flip
rate next to the nominal 7% rather than just asserting the nominal rate.

Usage:
  python route_and_review.py --db data/traces.duckdb --route-pct 18 --error-rate 0.07 \
      --out-dir data/filtered --calibration-n 300 --calibration-seed 99
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_problems import generate_reasoning_problems, make_cot_prompt
from review_sim import simulate_review
from store import connect, defect_rate, fetch_all, select_bottom_confidence_ids


def write_training_set(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/traces.duckdb")
    ap.add_argument("--route-pct", type=float, default=18.0)
    ap.add_argument("--error-rate", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out-dir", default="data/filtered")
    ap.add_argument("--calibration-n", type=int, default=300)
    ap.add_argument("--calibration-seed", type=int, default=99)
    ap.add_argument("--summary-out", default="docs/routing_summary.json")
    args = ap.parse_args()

    con = connect(args.db)
    all_records = fetch_all(con)
    n = len(all_records)
    k = round(n * args.route_pct / 100.0)

    routed_ids = set(select_bottom_confidence_ids(con, k))
    true_defective_ids = {r["id"] for r in all_records if r["is_defective"]}

    rng = random.Random(args.seed)

    # Budget arm: review only the routed slice.
    budget_removed_ids = set()
    for r in all_records:
        if r["id"] in routed_ids:
            flagged = simulate_review(r["is_defective"], rng, args.error_rate)
            if flagged:
                budget_removed_ids.add(r["id"])

    # Ceiling arm: review everything with the same simulated annotator.
    rng_ceiling = random.Random(args.seed + 1)
    ceiling_removed_ids = set()
    for r in all_records:
        flagged = simulate_review(r["is_defective"], rng_ceiling, args.error_rate)
        if flagged:
            ceiling_removed_ids.add(r["id"])

    unfiltered_set = all_records
    budget_set = [r for r in all_records if r["id"] not in budget_removed_ids]
    ceiling_set = [r for r in all_records if r["id"] not in ceiling_removed_ids]

    out_dir = Path(args.out_dir)
    write_training_set(out_dir / "unfiltered.jsonl", unfiltered_set)
    write_training_set(out_dir / "budget.jsonl", budget_set)
    write_training_set(out_dir / "ceiling.jsonl", ceiling_set)

    defects_in_routed = len(routed_ids & true_defective_ids)
    caught_of_routed_defects = len(budget_removed_ids & true_defective_ids & routed_ids)
    overall_catch_rate = (
        100.0 * caught_of_routed_defects / len(true_defective_ids) if true_defective_ids else 0.0
    )
    routed_precision = 100.0 * defects_in_routed / k if k else 0.0

    # 300-trace calibration: independent of the main dataset, measures the
    # simulator's realized flip rate against its own ground truth.
    calib_rng = random.Random(args.calibration_seed)
    calib_problems = generate_reasoning_problems(args.calibration_n, seed=args.calibration_seed + 1000)
    true_labels = [calib_rng.random() < 0.3 for _ in calib_problems]  # synthetic ground truth labels
    flips = 0
    for label in true_labels:
        reviewed = simulate_review(label, calib_rng, args.error_rate)
        if reviewed != label:
            flips += 1
    realized_error_rate = flips / len(true_labels)

    summary = {
        "n_total": n,
        "route_pct": args.route_pct,
        "k_routed": k,
        "defect_rate_pct": defect_rate(con),
        "true_defective_count": len(true_defective_ids),
        "defects_in_routed_slice": defects_in_routed,
        "routed_slice_precision_pct": routed_precision,
        "caught_of_total_defects_pct": overall_catch_rate,
        "budget_training_set_size": len(budget_set),
        "ceiling_training_set_size": len(ceiling_set),
        "unfiltered_training_set_size": len(unfiltered_set),
        "nominal_annotator_error_rate": args.error_rate,
        "calibration_n": args.calibration_n,
        "calibration_realized_error_rate": realized_error_rate,
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
