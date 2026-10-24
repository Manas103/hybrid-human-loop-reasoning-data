# Hybrid Human-in-the-Loop Synthetic Reasoning Data

Generates chain-of-thought traces for a two-step arithmetic task, flags the
worst-confidence slice for simulated human review, and measures how much a
downstream fine-tune recovers by spending that review budget instead of
either reviewing nothing or reviewing everything. Every number below was
measured on this machine, not targeted: trace generation, defect detection,
review routing, and all three fine-tunes ran locally, no API calls.

## Why this exists

A data engine that generates more synthetic reasoning data than any human
team can check by hand needs a way to spend a limited review budget where it
helps most. This project builds the smallest honest version of that pipeline:
generate traces, score each one's actual correctness and faithfulness (which
the router is not allowed to see), route the least-confident slice to a
simulated noisy reviewer, and fine-tune on what survives.

## Honest framing

- **What this is not.** Not a human annotation study; "review" is a seeded
  noise model (`src/review_sim.py`) that flips a trace's true defective/clean
  label with a stated error rate, applied to the generator's own ground
  truth, not to a human's judgment. This is stated on the page, not a
  footnote.
- **Scale was cut from the resume's 20,000 traces, and this is a compute and
  time budget decision for a single machine with one consumer GPU, not a
  measurement one.** 250 traces were generated; the
  defect-detection and routing mechanism does not change with scale, only the
  statistical precision of the measured rates does.
- **"Unfaithful step" is checkable only because the generator, not the model,
  knows the true intermediate value** (`src/reasoning_problems.py`): every
  problem is a two-step synthetic arithmetic word problem with a known answer
  after step 1, so a trace can get the final number right by compensating
  errors and still be caught as unfaithful. `src/defects.py` checks both
  independently, sharing no detection logic with the router's confidence
  signal.
- **The router's signal is the model's own generation confidence (mean
  per-token log-probability), not a peek at the ground truth.** A real review
  router cannot see which traces are actually wrong; it can only see signals
  available at generation time. Confidence is the honest version of that
  signal.
- **DuckDB is the router's selection mechanism, not a passive store.** The
  bottom-18%-by-confidence slice is a single `ORDER BY confidence ASC LIMIT k`
  query (`src/store.py`), not a hand-rolled sort, so the selection is
  inspectable with any SQL client against `data/traces.duckdb`.
- **Machine and toolchain, exactly as measured.** 8 physical / 16 logical
  cores (AMD Ryzen 7 7800X3D), Windows 11 Home (build 10.0.26200), WSL2
  Ubuntu 22.04 (capped at 12 cores / 12 GB RAM), Python 3.10 (WSL) / 3.12
  (Windows), PyTorch 2.6.0+cu124, Transformers 5.20.2, TRL 1.14.2 (SFTTrainer
  with LoRA), DuckDB 1.5.6, pytest, RTX 4070 SUPER (sm_89), model
  Qwen/Qwen2.5-1.5B-Instruct, bf16.

## Architecture

```
src/
  reasoning_problems.py   generates two-step synthetic word problems with a
                          known true intermediate value and true final answer
  defects.py               parses a trace's "Step 1 result" / "Answer" lines
                           and flags wrong_answer and unfaithful_step
                           independently
  review_sim.py             a seeded noisy labeler: flips a true label with
                            probability error_rate
  store.py                  DuckDB schema, insert, and the router's bottom-k
                            confidence selection query
  model_io.py                loads Qwen2.5-1.5B-Instruct and generates with
                            per-token confidence via compute_transition_scores
scripts/
  generate_traces.py         chunked, resumable trace generation; writes to
                            DuckDB once a full run completes
  route_and_review.py        computes the routed slice, simulates review for
                            the budget and ceiling arms, writes the three
                            training sets, and runs the 300-trace calibration
  sft_finetune.py             LoRA SFT on one filtered training set
                            (TRL SFTTrainer, completion_only_loss)
  eval_sft.py                 chunked, resumable task-accuracy eval on a
                            held-out test split (disjoint generation seed)
tests/
  test_reasoning_problems.py, test_defects.py, test_review_sim.py, test_store.py
docs/
  trace_generation_output.txt, routing_summary.json, sft_eval_output.txt
data/
  traces.duckdb (gitignored; regenerate with generate_traces.py)
```

**Why defects.py and the router's confidence signal are independent code
paths.** If the router used the same logic that defines a defect, routing
would trivially achieve 100% precision by construction and the experiment
would measure nothing; confidence is a signal available without knowing the
true label, which is the realistic constraint a real review system faces.

## Validation

**1. Test suite** (`docs/test_output.txt`):

```
................
16 passed in 1.01s
```

**2. Trace generation and routing** (`docs/routing_summary.json`):

```json
{
  "n_total": 250,
  "route_pct": 18.0,
  "k_routed": 45,
  "defect_rate_pct": 51.6,
  "true_defective_count": 129,
  "defects_in_routed_slice": 30,
  "routed_slice_precision_pct": 66.67,
  "caught_of_total_defects_pct": 21.71,
  "budget_training_set_size": 221,
  "ceiling_training_set_size": 117,
  "unfiltered_training_set_size": 250,
  "nominal_annotator_error_rate": 0.07,
  "calibration_n": 300,
  "calibration_realized_error_rate": 0.0733
}
```

**3. Downstream fine-tune comparison** (`docs/eval_sft_*.json`), task accuracy
on a 60-problem held-out test set (disjoint generation seed):

| Arm | Task accuracy | Faithful-step rate |
|---|---|---|
| base (no fine-tune) | 91.67% | 61.67% |
| unfiltered (all 250 traces) | 90.00% | 80.00% |
| budget (confidence-routed 18%, reviewed) | 88.33% | 76.67% |
| ceiling (everything reviewed) | 86.67% | 66.67% |

## Findings

**The first prompt design produced a 100% defect rate, and the bug was in
the prompt, not the model.** The original two-line "Step 1 result" / "Step 2
result" prompt did not say which sub-computation counted as step 1, so the
model consistently folded both arithmetic operations into what it called
step 2 and reported the plain starting quantity as "step 1," a different but
internally valid decomposition from the one `step1_true` assumed. Every one
of the first 400 traces was flagged `unfaithful_step` as a result, which is a
prompt-ambiguity artifact, not evidence the model's reasoning was wrong. The
fix was making step 1 explicit ("the amount right after the delivery
arrives, before anything ships out"); after the fix, the defect rate dropped
to a believable 51.6% on a fresh 250-trace generation. This is reported
instead of quietly keeping the first run, per this portfolio's practice of
disclosing a first measurement that was wrong and the fix.

**The three-way fine-tune comparison came out in the opposite order from the
resume's claim, and the honest explanation is a quantity/quality tradeoff,
not a broken pipeline.** Measured task accuracy was unfiltered (90.00%) >
budget (88.33%) > ceiling (86.67%), all three below the un-fine-tuned base
model (91.67%). The ceiling arm removes the most training examples (down to
117 of 250, because the simulated reviewer sees every trace and the true
defect rate is 51.6%), the budget arm removes fewer (down to 221, because
only the routed 18% gets reviewed), and unfiltered keeps all 250. At this
scale, less data hurt more than including defective traces helped: the
"highest-quality" training set was also the smallest one, and training-set
size dominated the result. This does not mean review-based filtering cannot
help, it means that at 250 traces and one epoch of LoRA SFT on a 1.5B model,
the quantity effect was larger than the quality effect, which is itself a
finding worth having.

**The routed slice has real precision (66.7% of routed items are actually
defective, against the 51.6% base rate) but catches only 21.7% of all
defects, because only 18% of traces are ever reviewed.** Generation
confidence concentrates defects somewhat above their base rate but is far
from a perfect defect detector; a budget of 18% structurally cannot catch
more than 18% of a uniformly-distributed defect population regardless of
how good the ranking is, and 21.7% shows the ranking is doing a little
better than that floor, not dramatically better.

## Measured results

Machine: 8 physical / 16 logical cores (AMD Ryzen 7 7800X3D), Windows 11 Home
(build 10.0.26200) with WSL2 Ubuntu 22.04 (12 cores / 12 GB RAM), RTX 4070
SUPER, CUDA 12.4, Qwen2.5-1.5B-Instruct, bf16, LoRA rank 16.

**Headline: fine-tuning on any of the three filtered sets did not beat the
un-fine-tuned base model at this scale (base 91.67% vs. unfiltered 90.00%,
budget 88.33%, ceiling 86.67%), and the three fine-tuned arms ranked by
training-set size, not by review quality, because the review-everything
ceiling arm also had the fewest surviving examples (117 of 250).**

| Claim | Measured | Meets claim |
|---|---|---|
| 20,000 generated traces | 250 (budget-reduced; see Honest framing) | no, by design |
| 31% carried a wrong answer or unfaithful step | 51.6% (after fixing a prompt-ambiguity bug that initially caused 100%, see Findings) | no, higher |
| router sent the worst 18% to review | 45/250 routed (18.0%) | yes |
| caught 76% of the defective traces | 21.7% | no |
| fine-tuning on that budget landed 1.2 points under the ceiling | budget (88.33%) landed 1.67 points *above* the ceiling arm (86.67%), opposite direction | no |
| 6.4 points above unfiltered | budget landed 1.67 points *below* unfiltered (90.00%), opposite direction | no |
| review simulated at a 7% annotator error rate | implemented; realized rate on 300-trace calibration: 7.33% | yes |
| 300 traces hand-verified | 300-trace **simulated** calibration check, not hand-verified (see Honest framing) | no, disclosed |

## Building and running

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

Trace generation, fine-tuning, and evaluation require the GPU-side venv
(WSL2 Ubuntu 22.04 with the packages in `requirements.txt` installed against
CUDA 12.4 torch):

```
python scripts/generate_traces.py --n <N> --seed 30 --out-db data/traces.duckdb
python scripts/route_and_review.py --db data/traces.duckdb --out-dir data/filtered
python scripts/sft_finetune.py --train-jsonl data/filtered/unfiltered.jsonl --out-dir runs/sft_unfiltered
python scripts/sft_finetune.py --train-jsonl data/filtered/budget.jsonl --out-dir runs/sft_budget
python scripts/sft_finetune.py --train-jsonl data/filtered/ceiling.jsonl --out-dir runs/sft_ceiling
python scripts/eval_sft.py --adapter runs/sft_unfiltered/adapter --out docs/eval_sft_unfiltered.json
python scripts/eval_sft.py --adapter runs/sft_budget/adapter --out docs/eval_sft_budget.json
python scripts/eval_sft.py --adapter runs/sft_ceiling/adapter --out docs/eval_sft_ceiling.json
```

`generate_traces.py` and `eval_sft.py` are chunked and resumable: rerun the
same command if it stops partway, it picks up where it left off.

## Sibling comparison

This portfolio's `note-quality-judge-harness`
(https://github.com/Manas103/note-quality-judge-harness) also measures a gap
between what a cheap signal can tell you and what the ground truth actually
is (there, a reference-free judge's accuracy over-credit; here, a confidence
score's routing precision). That project has no review budget or downstream
training step; this one's contribution is specifically the budget-allocation
question, how much of a fixed review budget to spend and where.

## Limitations

- Scale is reduced from the resume's 20,000 traces; see Honest framing.
- "Review" is a seeded noise model applied to the generator's own ground
  truth, not real human annotation; the 7% error rate is an assumption, not a
  measured property of any real annotator pool.
- The confidence router uses the same model that generated the traces; a
  production system might use a separate, cheaper scoring model, which could
  have different (better or worse) routing precision than measured here.
- Defect detection requires the strict three-line output format
  (`src/defects.py`); a response that doesn't follow the format at all is
  scored as a wrong answer, which may overstate the wrong-answer rate
  relative to a looser parser.
