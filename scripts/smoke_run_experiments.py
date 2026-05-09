#!/usr/bin/env python3
"""
Smoke test for scripts/run_experiments.py.

Verifies the multi-run launcher end-to-end without calling any LLM, Lean
REPL, or external service.  Works by pre-populating seed output directories
with valid JSONL results so the launcher's skip-detection fires immediately
and only the load -> evaluate -> aggregate -> save path is exercised.

What this covers
----------------
- SeedConfig seed generation
- is_completed() -> True (completed seed skipped, no subprocess spawned)
- _pipeline_metrics() -> importlib load of run.py + Evaluator.evaluate()
- MultiSeedAggregator.aggregate() -> mean / std / 95% CI
- _save_summary() -> per_seed.csv, aggregate.csv, summary.md
- CLI main() round-trip

What this does NOT cover
------------------------
- Subprocess launch (_execute_seed) — tested by unit tests
- Real LLM / Lean calls — tested by the main smoke test (make smoke)

Usage
-----
    python3 scripts/smoke_run_experiments.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_run_experiments")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Fake JSONL records — valid PipelineResult JSON understood by _load_pipeline_results
# ---------------------------------------------------------------------------

def _proved_record(idx: int, seed: int) -> dict:
    return {
        "conjecture": {
            "id": f"smoke_s{seed}_{idx}",
            "domain": "number_theory",
            "nl_statement": f"For all n, {idx}*(n+1) has property P",
            "variables": ["n"],
            "source": "smoke",
            "timestamp": "2026-05-10T00:00:00+00:00",
            "metadata": {},
        },
        "filter_result": {
            "status": "surviving",
            "counterexample": None,
            "numerical_evidence": {"n=0": "0", "n=1": "2"},
        },
        "formalization": {
            "lean_code": "theorem t (n : ℕ) : True := trivial",
            "status": "typechecks",
            "retries": 0,
            "error_history": [],
        },
        "proof": {
            "status": "proved",
            "proof": "theorem t (n : ℕ) : True := trivial",
            "layer": "layer0_auto",
            "attempts": [
                {"tactic": "trivial", "success": True, "error": None, "layer": "layer0_auto"}
            ],
            "duration_seconds": 0.002,
        },
        "final_status": "proved",
    }


def _filtered_record(idx: int, seed: int) -> dict:
    return {
        "conjecture": {
            "id": f"smoke_s{seed}_{idx}_filtered",
            "domain": "number_theory",
            "nl_statement": f"For all n, n > n+1 (false)",
            "variables": ["n"],
            "source": "smoke",
            "timestamp": "2026-05-10T00:00:00+00:00",
            "metadata": {},
        },
        "filter_result": {
            "status": "disproved",
            "counterexample": {"n": 0},
            "numerical_evidence": {},
        },
        "formalization": None,
        "proof": None,
        "final_status": "filtered_out",
    }


def _write_seed_results(seed_dir: Path, seed: int) -> None:
    """Write 3 fake pipeline results: 2 proved, 1 filtered out."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _proved_record(0, seed),
        _proved_record(1, seed),
        _filtered_record(2, seed),
    ]
    with (seed_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    logger.info("  wrote %d records -> %s/results.jsonl", len(records), seed_dir.name)


# ---------------------------------------------------------------------------
# Load run_experiments module
# ---------------------------------------------------------------------------

def _load_run_experiments():
    script = _REPO_ROOT / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 64)
    print("  run_experiments.py smoke test")
    print("  (no LLM, no Lean, no GPU — pre-populated seed dirs)")
    print("=" * 64)

    # -- Step 1: compute seeds ---------------------------------------------
    print("\nStep 1/4  Computing seeds for n_seeds=2, base_seed=42 ...")
    from conjlean.multi_seed import SeedConfig
    sc = SeedConfig(n_seeds=2, base_seed=42)
    print(f"          Seeds: {sc.seeds}")

    # -- Step 2: populate output directory ---------------------------------
    print("\nStep 2/4  Pre-populating seed directories with fake JSONL ...")
    tmp = tempfile.mkdtemp(prefix="conjlean_smoke_exp_")
    output_dir = Path(tmp)
    for seed in sc.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        _write_seed_results(seed_dir, seed)
    print(f"          Output root: {output_dir}")

    # -- Step 3: run launcher ----------------------------------------------
    print("\nStep 3/4  Running run_experiments.py main() ...")
    print("          (all seeds already completed -> no subprocess launched)")
    print()

    mod = _load_run_experiments()
    ret = mod.main([
        "--type", "pipeline",
        "--config", str(_REPO_ROOT / "configs" / "config.yaml"),
        "--n-seeds", "2",
        "--base-seed", "42",
        "--output-dir", str(output_dir),
        "--log-level", "INFO",
    ])

    if ret != 0:
        print(f"\nFAIL — main() returned {ret}")
        return 1

    # -- Step 4: inspect generated files ----------------------------------
    print("\nStep 4/4  Generated files:")
    print()
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            rel = f.relative_to(output_dir)
            print(f"  {rel}  ({size} bytes)")

    agg_csv = output_dir / "multi_seed_pipeline_aggregate.csv"
    summary_md = output_dir / "multi_seed_pipeline_summary.md"

    print()
    if agg_csv.exists():
        print("-- aggregate CSV ----------------------------------------------")
        print(agg_csv.read_text(encoding="utf-8"))

    if summary_md.exists():
        print("-- summary markdown -------------------------------------------")
        print(summary_md.read_text(encoding="utf-8"))

    # -- Assertions ---------------------------------------------------------
    failures = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"  OK  {msg}")
        else:
            print(f"  FAIL  FAIL: {msg}")
            failures.append(msg)

    print("Assertions:")
    check(agg_csv.exists(), "aggregate CSV exists")
    check(summary_md.exists(), "summary markdown exists")
    check((output_dir / "multi_seed_pipeline_per_seed.csv").exists(), "per-seed CSV exists")

    if agg_csv.exists():
        text = agg_csv.read_text()
        check("end_to_end_rate" in text, "aggregate CSV contains end_to_end_rate")
        check("proof_search_rate" in text, "aggregate CSV contains proof_search_rate")

    if summary_md.exists():
        text = summary_md.read_text()
        check("## Aggregate" in text, "markdown has Aggregate section")
        check("## Per-Seed" in text, "markdown has Per-seed section")

    print()
    print("=" * 64)
    if failures:
        print(f"  FAIL — {len(failures)} assertion(s) failed")
        print("=" * 64)
        return 1

    print("  PASS — multi-run launcher smoke test complete")
    print("=" * 64)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
