"""
Multi-seed experiment infrastructure for statistically robust paper reporting.

Run the pipeline or REFUTE loop N times under different random seeds, collect
per-run metric dictionaries, and aggregate them into mean ± std ± 95% CI.

Typical usage::

    from conjlean.multi_seed import (
        SeedConfig,
        MultiSeedAggregator,
        SeedRunMetrics,
        metrics_from_evaluation_report,
        metrics_from_refute_metrics,
    )

    cfg = SeedConfig(n_seeds=5, base_seed=0)
    print(cfg.seeds)  # [842302851, 471086878, ...]

    # After running the pipeline N times and evaluating each result set:
    runs = [
        metrics_from_evaluation_report(report_i, seed=cfg.seeds[i], run_index=i)
        for i, report_i in enumerate(reports)
    ]
    agg = MultiSeedAggregator().aggregate(runs)
    MultiSeedAggregator().save_all(agg, Path("data/results/multi_seed"))
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from conjlean.evaluate import EvaluationReport
    from conjlean.refute_evaluate import RefuteMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CI_LEVEL = 0.95

# Two-tailed t critical values at α = 0.025 (95% CI), indexed by degrees of
# freedom.  Scipy is used when available; this table is the fallback.
_T_CRIT_TABLE: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _t_critical(df: int, ci_level: float = _DEFAULT_CI_LEVEL) -> float:
    """Return the two-tailed t critical value for ``df`` degrees of freedom."""
    try:
        from scipy.stats import t as scipy_t  # type: ignore[import]
        return float(scipy_t.ppf((1.0 + ci_level) / 2.0, df=df))
    except ImportError:
        pass

    # Fallback: exact table for small df, z-approximation for df > 30.
    if ci_level != 0.95:
        # Table only covers 95%; warn and use 1.96 for other levels
        logger.warning(
            "scipy not installed; ci_level=%.2f not supported in fallback — "
            "using 1.96 (z-approximation, valid only for large n).",
            ci_level,
        )
        return 1.960

    if df > 30:
        return 1.960

    # Return value for exact df, or next-lower tabulated df (conservative).
    if df in _T_CRIT_TABLE:
        return _T_CRIT_TABLE[df]
    for k in sorted(_T_CRIT_TABLE.keys(), reverse=True):
        if k < df:
            return _T_CRIT_TABLE[k]
    return _T_CRIT_TABLE[1]


def _compute_ci(
    values: list[float],
    ci_level: float = _DEFAULT_CI_LEVEL,
) -> "tuple[float, float] | None":
    """t-based CI on the mean. Returns ``None`` when n < 2."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = variance ** 0.5
    se = std / (n ** 0.5)
    t_crit = _t_critical(df=n - 1, ci_level=ci_level)
    half = t_crit * se
    return (mean - half, mean + half)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SeedConfig:
    """
    Deterministic seed list derived from a single base seed.

    Attributes:
        n_seeds: Number of independent runs to generate seeds for.
        base_seed: Master seed that controls the seed list.  Changing this
            produces a completely different (but equally reproducible) list.
    """

    n_seeds: int
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if self.base_seed < 0:
            raise ValueError(f"base_seed must be >= 0, got {self.base_seed}")

    @property
    def seeds(self) -> list[int]:
        """Reproducible list of ``n_seeds`` non-negative integer seeds."""
        import numpy as np

        rng = np.random.default_rng(self.base_seed)
        return [int(x) for x in rng.integers(0, 2**31, size=self.n_seeds)]


@dataclass
class SeedRunMetrics:
    """
    Flat metric dictionary from a single seed run.

    Attributes:
        seed: The random seed used for this run.
        run_index: Zero-based position in the seed list.
        metrics: Mapping of metric name → scalar float value.  All pipeline
            stages and REFUTE metrics should appear as separate keys.
    """

    seed: int
    run_index: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """
    Aggregated statistics across all seed runs.

    Attributes:
        n_seeds: Number of independent runs that were aggregated.
        seeds: Ordered list of seeds used.
        metric_names: Sorted list of metric keys present in all runs.
        mean: Per-metric arithmetic mean.
        std: Per-metric sample standard deviation (ddof=1).
        ci_lower: Lower bound of the 95% t-CI on the mean.
        ci_upper: Upper bound of the 95% t-CI on the mean.
        ci_level: Confidence level used (default 0.95).
        runs: Raw per-seed results (preserved for CSV export).
    """

    n_seeds: int
    seeds: list[int]
    metric_names: list[str]
    mean: dict[str, float]
    std: dict[str, float]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    ci_level: float
    runs: list[SeedRunMetrics]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class MultiSeedAggregator:
    """
    Aggregates per-seed :class:`SeedRunMetrics` into :class:`AggregatedMetrics`
    and exports CSV and Markdown reports.
    """

    def aggregate(
        self,
        runs: list[SeedRunMetrics],
        *,
        ci_level: float = _DEFAULT_CI_LEVEL,
    ) -> AggregatedMetrics:
        """
        Compute mean, std, and t-based CI across all seed runs.

        Args:
            runs: One :class:`SeedRunMetrics` per seed.  All runs must share
                the same set of metric keys.
            ci_level: Confidence level for the interval (default 0.95).

        Returns:
            Populated :class:`AggregatedMetrics`.

        Raises:
            ValueError: If ``runs`` is empty or metric keys are inconsistent.
        """
        if not runs:
            raise ValueError("runs must not be empty.")

        all_keys: set[str] = set(runs[0].metrics.keys())
        for i, run in enumerate(runs[1:], start=1):
            if set(run.metrics.keys()) != all_keys:
                raise ValueError(
                    f"run[{i}] has different metric keys than run[0]: "
                    f"{sorted(set(run.metrics.keys()) ^ all_keys)}"
                )

        metric_names = sorted(all_keys)
        n = len(runs)
        seeds = [r.seed for r in runs]

        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        ci_lower: dict[str, float] = {}
        ci_upper: dict[str, float] = {}

        for key in metric_names:
            vals = [r.metrics[key] for r in runs]
            mu = sum(vals) / n
            mean[key] = round(mu, 8)
            if n >= 2:
                variance = sum((v - mu) ** 2 for v in vals) / (n - 1)
                sigma = variance ** 0.5
            else:
                sigma = 0.0
            std[key] = round(sigma, 8)
            ci = _compute_ci(vals, ci_level=ci_level)
            ci_lower[key] = round(ci[0], 8) if ci else mean[key]
            ci_upper[key] = round(ci[1], 8) if ci else mean[key]

        logger.info(
            "MultiSeedAggregator: aggregated %d runs over %d metrics.",
            n,
            len(metric_names),
        )
        return AggregatedMetrics(
            n_seeds=n,
            seeds=seeds,
            metric_names=metric_names,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            runs=runs,
        )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_markdown(self, aggregated: AggregatedMetrics) -> str:
        """
        Render the aggregated results as a Markdown document.

        The output contains two tables:

        1. **Aggregate Summary** — mean ± std and 95% CI per metric.
        2. **Per-Seed Results** — raw value per metric per seed run.

        Args:
            aggregated: Populated :class:`AggregatedMetrics`.

        Returns:
            Multi-section Markdown string.
        """
        ci_pct = int(round(aggregated.ci_level * 100))
        lines: list[str] = [
            f"# Multi-Seed Experiment Summary (N={aggregated.n_seeds} seeds)\n"
        ]

        # ── Aggregate summary table ──────────────────────────────────────────
        lines.append(f"## Aggregate Metrics ({ci_pct}% CI on the mean)\n")
        lines.append(f"| Metric | Mean | Std | {ci_pct}% CI |")
        lines.append("|---|---|---|---|")
        for key in aggregated.metric_names:
            lo = aggregated.ci_lower[key]
            hi = aggregated.ci_upper[key]
            lines.append(
                f"| {key} "
                f"| {aggregated.mean[key]:.4f} "
                f"| {aggregated.std[key]:.4f} "
                f"| [{lo:.4f}, {hi:.4f}] |"
            )
        lines.append("")

        # ── Per-seed table ───────────────────────────────────────────────────
        lines.append("## Per-Seed Results\n")
        header_cols = ["Seed", "Run"] + aggregated.metric_names
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "---|" * len(header_cols))
        for run in aggregated.runs:
            row = [str(run.seed), str(run.run_index)]
            row += [f"{run.metrics[k]:.4f}" for k in aggregated.metric_names]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # ── Seed list ────────────────────────────────────────────────────────
        lines.append("## Seed List\n")
        lines.append(
            "Seeds used (in order): "
            + ", ".join(str(s) for s in aggregated.seeds)
        )
        lines.append("")

        return "\n".join(lines)

    def to_csv(
        self, aggregated: AggregatedMetrics, path: Path
    ) -> tuple[Path, Path]:
        """
        Write two CSV files: one with per-seed rows and one with aggregate stats.

        Args:
            aggregated: Populated :class:`AggregatedMetrics`.
            path: Base path (without extension).  Two files are written:
                ``<path>_per_seed.csv`` and ``<path>_aggregate.csv``.

        Returns:
            Tuple of ``(per_seed_path, aggregate_path)``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        per_seed_path = path.parent / (path.name + "_per_seed.csv")
        aggregate_path = path.parent / (path.name + "_aggregate.csv")

        # Per-seed CSV
        with per_seed_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["seed", "run_index"] + aggregated.metric_names
            )
            writer.writeheader()
            for run in aggregated.runs:
                row: dict[str, object] = {
                    "seed": run.seed,
                    "run_index": run.run_index,
                }
                row.update(run.metrics)
                writer.writerow(row)

        # Aggregate CSV
        ci_col = f"ci{int(round(aggregated.ci_level * 100))}_lower"
        ci_col_hi = f"ci{int(round(aggregated.ci_level * 100))}_upper"
        with aggregate_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["stat"] + aggregated.metric_names,
            )
            writer.writeheader()
            writer.writerow({"stat": "mean", **aggregated.mean})
            writer.writerow({"stat": "std", **aggregated.std})
            writer.writerow({"stat": ci_col, **aggregated.ci_lower})
            writer.writerow({"stat": ci_col_hi, **aggregated.ci_upper})
            writer.writerow({"stat": "n_seeds", **{k: aggregated.n_seeds for k in aggregated.metric_names}})

        logger.info(
            "Wrote per-seed CSV to %s and aggregate CSV to %s",
            per_seed_path,
            aggregate_path,
        )
        return per_seed_path, aggregate_path

    def save_all(self, aggregated: AggregatedMetrics, base_path: Path) -> None:
        """
        Save all export formats: per-seed CSV, aggregate CSV, and Markdown.

        Args:
            aggregated: Populated :class:`AggregatedMetrics`.
            base_path: Base path without extension.  Created with parents as
                needed.  Writes ``<base>_per_seed.csv``, ``<base>_aggregate.csv``,
                and ``<base>_summary.md``.
        """
        base_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(aggregated, base_path)
        md_path = base_path.parent / (base_path.name + "_summary.md")
        md_path.write_text(self.to_markdown(aggregated), encoding="utf-8")
        logger.info("Multi-seed summary written to %s", md_path)


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------


def metrics_from_evaluation_report(
    report: "EvaluationReport",
    *,
    seed: int,
    run_index: int,
) -> SeedRunMetrics:
    """
    Extract a flat metric dict from an :class:`~conjlean.evaluate.EvaluationReport`.

    Extracted keys:

    - ``filtering_rate``, ``formalization_rate``, ``proof_search_rate``,
      ``end_to_end_rate`` — per-stage success rates.
    - ``filtering_n``, ``formalization_n``, ``proof_search_n``, ``end_to_end_n``
      — denominator counts.
    - ``mean_retries``, ``mean_proof_duration_s`` — timing / retry averages.

    Args:
        report: Populated :class:`~conjlean.evaluate.EvaluationReport`.
        seed: Seed value used for this run.
        run_index: Zero-based run position.

    Returns:
        :class:`SeedRunMetrics` with the extracted flat metric dict.
    """
    return SeedRunMetrics(
        seed=seed,
        run_index=run_index,
        metrics={
            "filtering_rate": report.filtering.rate,
            "formalization_rate": report.formalization.rate,
            "proof_search_rate": report.proof_search.rate,
            "end_to_end_rate": report.end_to_end.rate,
            "filtering_n": float(report.filtering.total),
            "formalization_n": float(report.formalization.total),
            "proof_search_n": float(report.proof_search.total),
            "end_to_end_n": float(report.end_to_end.total),
            "mean_retries": report.mean_formalization_retries,
            "mean_proof_duration_s": report.mean_proof_duration_seconds,
        },
    )


def metrics_from_refute_metrics(
    metrics: "RefuteMetrics",
    *,
    seed: int,
    run_index: int,
) -> SeedRunMetrics:
    """
    Extract a flat metric dict from a :class:`~conjlean.refute_evaluate.RefuteMetrics`.

    Extracted keys:

    - ``precision``, ``recall``, ``f1``, ``false_positive_rate``
    - ``n_total``, ``n_refuted``, ``n_survived``, ``n_refined``
    - ``refinement_rate``, ``mean_rounds``, ``mean_wall_seconds``

    Args:
        metrics: Populated :class:`~conjlean.refute_evaluate.RefuteMetrics`.
        seed: Seed value used for this run.
        run_index: Zero-based run position.

    Returns:
        :class:`SeedRunMetrics` with the extracted flat metric dict.
    """
    return SeedRunMetrics(
        seed=seed,
        run_index=run_index,
        metrics={
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "false_positive_rate": metrics.false_positive_rate,
            "n_total": float(metrics.n_total),
            "n_refuted": float(metrics.n_refuted),
            "n_survived": float(metrics.n_survived),
            "n_refined": float(metrics.n_refined),
            "refinement_rate": metrics.refinement_rate,
            "mean_rounds": metrics.mean_rounds,
            "mean_wall_seconds": metrics.mean_wall_seconds_per_conjecture,
        },
    )
