"""
Aggregate evaluation results across multiple seed runs.

Reads per-run result JSONL files (ConjLean pipeline or REFUTE loop), extracts
scalar metrics from each run, and outputs:

  - <output>_per_seed.csv     — one row per seed run
  - <output>_aggregate.csv    — mean / std / CI rows
  - <output>_summary.md       — human-readable markdown tables

Usage examples::

    # ConjLean pipeline results from 5 seed runs
    python3 scripts/run_multi_seed_aggregate.py \\
        --type pipeline \\
        --result-files data/results/seed_0/results.jsonl \\
                       data/results/seed_1/results.jsonl \\
                       data/results/seed_2/results.jsonl \\
        --seeds 842302851 471086878 1234567 \\
        --output data/results/multi_seed

    # REFUTE loop results, seeds inferred automatically from SeedConfig
    python3 scripts/run_multi_seed_aggregate.py \\
        --type refute \\
        --result-files data/refute_results/seed_*/loop_results.jsonl \\
        --benchmark-dir data/benchmark \\
        --n-seeds 5 \\
        --base-seed 0 \\
        --output data/results/multi_seed_refute
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result loader (mirrors run.py _load_pipeline_results)
# ---------------------------------------------------------------------------


def _load_pipeline_results(path: Path) -> list:
    from conjlean.schemas import (
        Conjecture,
        Domain,
        FilterResult,
        FilterStatus,
        FormalizedConjecture,
        FormalizationStatus,
        PipelineResult,
        PipelineStatus,
        ProofAttempt,
        ProofLayer,
        ProofResult,
        ProofStatus,
    )

    results: list[PipelineResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conjecture = Conjecture(
                id=record["conjecture"]["id"],
                domain=Domain(record["conjecture"]["domain"]),
                nl_statement=record["conjecture"]["nl_statement"],
                variables=record["conjecture"].get("variables", []),
            )
            filter_result = None
            if record.get("filter_result"):
                fr = record["filter_result"]
                filter_result = FilterResult(
                    conjecture=conjecture,
                    status=FilterStatus(fr["status"]),
                    counterexample=fr.get("counterexample"),
                    numerical_evidence=fr.get("numerical_evidence", {}),
                )
            formalization = None
            if record.get("formalization"):
                fc = record["formalization"]
                formalization = FormalizedConjecture(
                    conjecture=conjecture,
                    lean_code=fc["lean_code"],
                    status=FormalizationStatus(fc["status"]),
                    retries=fc.get("retries", 0),
                    error_history=fc.get("error_history", []),
                )
            proof = None
            if record.get("proof") and formalization is not None:
                pr = record["proof"]
                proof = ProofResult(
                    formalized=formalization,
                    status=ProofStatus(pr["status"]),
                    proof=pr.get("proof"),
                    layer=ProofLayer(pr["layer"]) if pr.get("layer") else None,
                    duration_seconds=pr.get("duration_seconds", 0.0),
                )
            results.append(
                PipelineResult(
                    conjecture=conjecture,
                    filter_result=filter_result,
                    formalization=formalization,
                    proof=proof,
                    final_status=PipelineStatus(record.get("final_status", "filtered_out")),
                )
            )
    return results


def _load_refute_results(path: Path) -> list:
    from conjlean.schemas import (
        Conjecture,
        Domain,
        RefuteLoopResult,
        RefuteLoopStatus,
    )

    results: list[RefuteLoopResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            orig = record["original_conjecture"]
            conjecture = Conjecture(
                id=orig["id"],
                domain=Domain(orig["domain"]),
                nl_statement=orig["nl_statement"],
                variables=orig.get("variables", []),
            )
            results.append(
                RefuteLoopResult(
                    original_conjecture=conjecture,
                    status=RefuteLoopStatus(record["status"]),
                    total_rounds=record.get("total_rounds", 0),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed evaluation results into CSV + Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type",
        choices=["pipeline", "refute"],
        required=True,
        help="Which result type to aggregate.",
    )
    parser.add_argument(
        "--result-files",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Paths to per-seed JSONL result files (one per seed run).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "Seed values corresponding to each result file.  If omitted, "
            "seeds are generated from --base-seed using SeedConfig."
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        metavar="N",
        help="Number of seeds (used with --base-seed to generate seed list).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        metavar="S",
        help="Base seed for deterministic seed list generation (default: 0).",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        metavar="DIR",
        help="Benchmark directory (required for --type refute).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Base output path (no extension).  CSV and MD files are appended.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        metavar="F",
        help="Confidence level (default: 0.95).",
    )
    parser.add_argument("--log-level", default="INFO", metavar="LEVEL")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    result_files = [Path(p) for p in args.result_files]
    n = len(result_files)

    # Determine seeds
    if args.seeds is not None:
        if len(args.seeds) != n:
            logger.error(
                "--seeds has %d values but %d result files were provided.",
                len(args.seeds),
                n,
            )
            return 1
        seeds = args.seeds
    else:
        from conjlean.multi_seed import SeedConfig
        n_seeds = args.n_seeds if args.n_seeds is not None else n
        if n_seeds != n:
            logger.warning(
                "--n-seeds (%d) differs from number of result files (%d); "
                "using first %d seeds.",
                n_seeds,
                n,
                n,
            )
        cfg = SeedConfig(n_seeds=max(n_seeds, n), base_seed=args.base_seed)
        seeds = cfg.seeds[:n]

    from conjlean.multi_seed import (
        MultiSeedAggregator,
        SeedRunMetrics,
        metrics_from_evaluation_report,
        metrics_from_refute_metrics,
    )

    runs: list[SeedRunMetrics] = []

    if args.type == "pipeline":
        from conjlean.evaluate import Evaluator

        for i, (path, seed) in enumerate(zip(result_files, seeds)):
            logger.info("[%d/%d] Loading pipeline results from %s", i + 1, n, path)
            pipeline_results = _load_pipeline_results(path)
            report = Evaluator().evaluate(pipeline_results)
            runs.append(
                metrics_from_evaluation_report(report, seed=seed, run_index=i)
            )

    elif args.type == "refute":
        if args.benchmark_dir is None:
            logger.error("--benchmark-dir is required for --type refute.")
            return 1

        from conjlean.benchmark import BenchmarkLoader
        from conjlean.refute_evaluate import RefuteEvaluator

        loader = BenchmarkLoader()
        benchmark_entries = loader.load_all(Path(args.benchmark_dir))
        logger.info("Loaded %d benchmark entries.", len(benchmark_entries))

        for i, (path, seed) in enumerate(zip(result_files, seeds)):
            logger.info("[%d/%d] Loading REFUTE results from %s", i + 1, n, path)
            loop_results = _load_refute_results(path)
            entries_slice = benchmark_entries[: len(loop_results)]
            metrics = RefuteEvaluator().evaluate(loop_results, entries_slice)
            runs.append(
                metrics_from_refute_metrics(metrics, seed=seed, run_index=i)
            )

    agg = MultiSeedAggregator().aggregate(runs, ci_level=args.ci_level)

    # Print summary to stdout
    print(MultiSeedAggregator().to_markdown(agg))

    # Save files
    base_path = Path(args.output)
    MultiSeedAggregator().save_all(agg, base_path)

    logger.info(
        "Saved: %s_per_seed.csv, %s_aggregate.csv, %s_summary.md",
        base_path.name,
        base_path.name,
        base_path.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
