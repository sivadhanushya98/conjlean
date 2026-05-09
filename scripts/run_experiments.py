"""
Multi-run experiment launcher for statistically robust paper results.

Executes the ConjLean pipeline or REFUTE loop N times, one run per seed,
writing each run to its own output directory.  After all runs complete,
aggregates per-seed metrics into mean / std / 95% CI using the existing
MultiSeedAggregator infrastructure.

Resumable: seeds whose output directory already contains a non-empty result
file are skipped automatically.  Re-run the same command to pick up from
where a previous invocation stopped.

Honesty note on randomness
--------------------------
``--seed N`` sets Python's stdlib random module seed at startup of each child
process.  This weakly controls non-LLM random components (e.g. SymPy filter
test-value sampling).  It does NOT make runs reproducible in a strong sense:

- LLM outputs (generation, formalization, proof dialogue) are non-deterministic
  regardless of the seed.
- The Refuter's RANDOM_STRUCTURED strategy uses ``random.Random()`` (OS-seeded)
  by design, not the global random state.
- The SymPy filter runs in subprocess workers with independent RNG state.

Multiple runs with different seeds therefore sample genuine LLM stochasticity,
which is exactly what is needed to estimate variance for paper reporting.

Usage examples::

    # Run 5 independent ConjLean pipeline experiments
    python3 scripts/run_experiments.py \\
        --type pipeline \\
        --config configs/config.yaml \\
        --domains number_theory inequality \\
        --n-per-domain 200 \\
        --n-seeds 5 \\
        --base-seed 0 \\
        --output-dir data/experiments/pipeline_v1

    # Run 5 independent REFUTE experiments (requires benchmark)
    python3 scripts/run_experiments.py \\
        --type refute \\
        --config configs/config.yaml \\
        --benchmark-dir data/benchmark \\
        --n-seeds 5 \\
        --base-seed 0 \\
        --output-dir data/experiments/refute_v1

    # Resume a partial run (already-completed seed dirs are skipped)
    python3 scripts/run_experiments.py \\
        --type pipeline \\
        --config configs/config.yaml \\
        --n-seeds 5 \\
        --output-dir data/experiments/pipeline_v1  # same dir as before
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the repo-root run.py, relative to this script.
_DEFAULT_RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    Parameters for a multi-run experiment.

    Attributes:
        experiment_type: ``"pipeline"`` or ``"refute"``.
        config: Path to the YAML config file passed to ``run.py``.
        output_dir: Root directory.  Per-seed subdirs are created inside.
        n_seeds: Number of independent runs.
        base_seed: Master seed for the deterministic seed list.
        run_py: Path to ``run.py``.  Defaults to ``<repo_root>/run.py``.
        domains: Domains to pass to ``--domains`` (pipeline only).
        n_per_domain: Conjectures per domain (pipeline only).
        provider: Optional LLM provider override.
        benchmark_dir: Benchmark directory (refute only, required).
        max_rounds: Max R-Agent rounds per conjecture (refute only).
        max_refinements: Max C-Agent refinements per conjecture (refute only).
        max_concurrent: Max concurrent conjectures (refute only).
    """

    experiment_type: str
    config: str
    output_dir: Path
    n_seeds: int
    base_seed: int = 0
    run_py: Path = field(default_factory=lambda: _DEFAULT_RUN_PY)
    domains: list[str] = field(default_factory=list)
    n_per_domain: int = 100
    provider: Optional[str] = None
    benchmark_dir: Optional[str] = None
    max_rounds: int = 10
    max_refinements: int = 3
    max_concurrent: int = 4

    def __post_init__(self) -> None:
        if self.experiment_type not in ("pipeline", "refute"):
            raise ValueError(
                f"experiment_type must be 'pipeline' or 'refute', got {self.experiment_type!r}"
            )
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if self.experiment_type == "refute" and not self.benchmark_dir:
            raise ValueError("benchmark_dir is required when experiment_type='refute'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """
    Orchestrates N independent pipeline or REFUTE runs and aggregates results.

    Each run is executed as a subprocess call to ``run.py`` so that the Lean
    REPL, LLM clients, and all Python state start completely fresh per seed.
    The subprocess receives ``--seed N`` which sets stdlib random in the child
    process (see module docstring for scope and limitations).
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        from conjlean.multi_seed import SeedConfig

        self._seed_cfg = SeedConfig(
            n_seeds=config.n_seeds, base_seed=config.base_seed
        )

    @property
    def seeds(self) -> list[int]:
        return self._seed_cfg.seeds

    def seed_dir(self, seed: int) -> Path:
        """Return the per-seed output directory for ``seed``."""
        return self.config.output_dir / f"seed_{seed}"

    def result_file(self, seed_dir: Path) -> Path:
        """Return the primary result file path inside ``seed_dir``."""
        if self.config.experiment_type == "pipeline":
            return seed_dir / "results.jsonl"
        return seed_dir / "loop_results.jsonl"

    def is_completed(self, seed: int) -> bool:
        """Return True if this seed's result file exists and is non-empty."""
        rf = self.result_file(self.seed_dir(seed))
        return rf.exists() and rf.stat().st_size > 0

    def build_command(self, seed: int, seed_dir: Path) -> list[str]:
        """Build the subprocess command for a single seed run."""
        cmd = [sys.executable, str(self.config.run_py)]
        if self.config.experiment_type == "pipeline":
            cmd += ["run", "--config", self.config.config]
            if self.config.provider:
                cmd += ["--provider", self.config.provider]
            if self.config.domains:
                cmd += ["--domains"] + self.config.domains
            cmd += ["--n-per-domain", str(self.config.n_per_domain)]
            cmd += ["--output", str(seed_dir)]
            cmd += ["--seed", str(seed)]
        else:
            cmd += ["refute", "--config", self.config.config]
            if self.config.provider:
                cmd += ["--provider", self.config.provider]
            cmd += ["--benchmark-dir", str(self.config.benchmark_dir)]
            cmd += ["--output", str(seed_dir)]
            cmd += ["--max-rounds", str(self.config.max_rounds)]
            cmd += ["--max-refinements", str(self.config.max_refinements)]
            cmd += ["--max-concurrent", str(self.config.max_concurrent)]
            cmd += ["--seed", str(seed)]
        return cmd

    def run_all(self) -> "AggregatedMetrics":
        """
        Execute all seed runs (skipping completed ones) and aggregate results.

        Returns:
            :class:`~conjlean.multi_seed.AggregatedMetrics` containing mean,
            std, and 95% CI for each metric across all seeds.
        """
        n = len(self.seeds)
        skipped = sum(1 for s in self.seeds if self.is_completed(s))
        logger.info(
            "ExperimentRunner | type=%s | n_seeds=%d | already_done=%d | to_run=%d",
            self.config.experiment_type,
            n,
            skipped,
            n - skipped,
        )

        seed_run_metrics = []
        for idx, seed in enumerate(self.seeds):
            sd = self.seed_dir(seed)
            if self.is_completed(seed):
                logger.info(
                    "[%d/%d] Seed %d: result file exists — skipping run.",
                    idx + 1, n, seed,
                )
            else:
                logger.info("[%d/%d] Seed %d: launching...", idx + 1, n, seed)
                self._execute_seed(seed, sd)

            metrics = self._load_run_metrics(idx, seed, sd)
            seed_run_metrics.append(metrics)

        from conjlean.multi_seed import MultiSeedAggregator

        agg = MultiSeedAggregator().aggregate(seed_run_metrics)
        self._save_summary(agg)
        return agg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_seed(self, seed: int, seed_dir: Path) -> None:
        """Run a single seed experiment as a subprocess."""
        seed_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(seed, seed_dir)
        logger.info("Command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _load_run_metrics(
        self, idx: int, seed: int, seed_dir: Path
    ) -> "SeedRunMetrics":
        """Load the result file for one seed and extract scalar metrics."""
        if self.config.experiment_type == "pipeline":
            return self._pipeline_metrics(idx, seed, seed_dir)
        return self._refute_metrics(idx, seed, seed_dir)

    def _pipeline_metrics(
        self, idx: int, seed: int, seed_dir: Path
    ) -> "SeedRunMetrics":
        from conjlean.evaluate import Evaluator
        from conjlean.multi_seed import metrics_from_evaluation_report

        # Import _load_pipeline_results from the repo-root run.py
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_run_module", self.config.run_py
        )
        run_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_mod)  # type: ignore[union-attr]

        results = run_mod._load_pipeline_results(self.result_file(seed_dir))
        report = Evaluator().evaluate(results)
        return metrics_from_evaluation_report(report, seed=seed, run_index=idx)

    def _refute_metrics(
        self, idx: int, seed: int, seed_dir: Path
    ) -> "SeedRunMetrics":
        from conjlean.benchmark import BenchmarkLoader
        from conjlean.refute_evaluate import RefuteEvaluator
        from conjlean.multi_seed import metrics_from_refute_metrics
        from conjlean.schemas import (
            Conjecture,
            Domain,
            RefuteLoopResult,
            RefuteLoopStatus,
        )

        # Load loop results
        loop_results: list[RefuteLoopResult] = []
        with self.result_file(seed_dir).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                orig = record["original_conjecture"]
                conj = Conjecture(
                    id=orig["id"],
                    domain=Domain(orig["domain"]),
                    nl_statement=orig["nl_statement"],
                    variables=orig.get("variables", []),
                )
                loop_results.append(
                    RefuteLoopResult(
                        original_conjecture=conj,
                        status=RefuteLoopStatus(record["status"]),
                        total_rounds=record.get("total_rounds", 0),
                    )
                )

        loader = BenchmarkLoader()
        entries = loader.load_all(Path(self.config.benchmark_dir))
        entries = entries[: len(loop_results)]

        metrics = RefuteEvaluator().evaluate(loop_results, entries)
        return metrics_from_refute_metrics(metrics, seed=seed, run_index=idx)

    def _save_summary(self, agg: "AggregatedMetrics") -> None:
        from conjlean.multi_seed import MultiSeedAggregator

        base = (
            self.config.output_dir
            / f"multi_seed_{self.config.experiment_type}"
        )
        MultiSeedAggregator().save_all(agg, base)
        logger.info(
            "Aggregate summary saved: %s_per_seed.csv, %s_aggregate.csv, %s_summary.md",
            base.name, base.name, base.name,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run N independent experiments and aggregate results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples")[1] if "Usage examples" in __doc__ else "",
    )
    parser.add_argument(
        "--type",
        choices=["pipeline", "refute"],
        required=True,
        dest="experiment_type",
        help="Which pipeline to run.",
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH", help="Path to YAML config file."
    )
    parser.add_argument(
        "--output-dir", required=True, metavar="DIR", help="Root output directory."
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5, metavar="N", help="Number of runs (default: 5)."
    )
    parser.add_argument(
        "--base-seed", type=int, default=0, metavar="S",
        help="Base seed for deterministic seed list (default: 0).",
    )
    # Pipeline options
    parser.add_argument(
        "--domains", nargs="+", default=["number_theory", "inequality"],
        metavar="DOMAIN", help="Domains to generate conjectures for (pipeline only).",
    )
    parser.add_argument(
        "--n-per-domain", type=int, default=100, metavar="N",
        help="Conjectures per domain (pipeline only).",
    )
    parser.add_argument(
        "--provider", default=None, metavar="NAME",
        help="Override LLM provider from config.",
    )
    # Refute options
    parser.add_argument(
        "--benchmark-dir", default=None, metavar="DIR",
        help="Benchmark directory (refute only, required for --type refute).",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=10, metavar="N",
        help="Max R-Agent rounds per conjecture (refute only).",
    )
    parser.add_argument(
        "--max-refinements", type=int, default=3, metavar="N",
        help="Max C-Agent refinements (refute only).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=4, metavar="N",
        help="Max concurrent conjectures (refute only).",
    )
    # Misc
    parser.add_argument(
        "--run-py", default=None, metavar="PATH",
        help="Path to run.py (default: auto-detected from this script's location).",
    )
    parser.add_argument("--log-level", default="INFO", metavar="LEVEL")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_py = Path(args.run_py) if args.run_py else _DEFAULT_RUN_PY

    try:
        exp_config = ExperimentConfig(
            experiment_type=args.experiment_type,
            config=args.config,
            output_dir=Path(args.output_dir),
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
            run_py=run_py,
            domains=args.domains,
            n_per_domain=args.n_per_domain,
            provider=args.provider,
            benchmark_dir=args.benchmark_dir,
            max_rounds=args.max_rounds,
            max_refinements=args.max_refinements,
            max_concurrent=args.max_concurrent,
        )
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1

    runner = ExperimentRunner(exp_config)

    logger.info(
        "Starting %s experiment | n_seeds=%d | base_seed=%d | output=%s",
        args.experiment_type,
        args.n_seeds,
        args.base_seed,
        args.output_dir,
    )
    logger.info(
        "Seeds: %s",
        ", ".join(str(s) for s in runner.seeds),
    )
    logger.info(
        "NOTE: --seed controls stdlib random only.  LLM outputs are "
        "non-deterministic; runs sample genuine stochastic variance.",
    )

    try:
        agg = runner.run_all()
    except subprocess.CalledProcessError as exc:
        logger.error("Subprocess failed (exit %d): %s", exc.returncode, exc.cmd)
        return exc.returncode
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    # Print markdown summary to stdout
    from conjlean.multi_seed import MultiSeedAggregator
    print(MultiSeedAggregator().to_markdown(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
