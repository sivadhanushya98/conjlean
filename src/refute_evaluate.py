"""
REFUTE-specific evaluation metrics for the ConjLean / REFUTE pipeline.

Distinct from the general :class:`~conjlean.evaluate.Evaluator`, which tracks
Lean proof success rates.  This module measures counterexample-finding
performance for the ICML AI4Research 2026 paper submission.

Metrics reported:

- Primary: precision, recall, F1, false-positive rate
- Secondary: per-strategy, per-domain, per-tier breakdowns;
  mean rounds to counterexample; refinement quality
- Ablation: configurable variants for Table 2 (no strategist, no refinement,
  no-LLM R-Agent)

Usage::

    evaluator = RefuteEvaluator()
    metrics = evaluator.evaluate(loop_results, benchmark_entries)
    evaluator.print_report(metrics)
    evaluator.save_report(metrics, Path("results/run_01"))
    print(evaluator.to_latex_table(ablation))
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from conjlean.schemas import (
    BenchmarkEntry,
    BenchmarkTier,
    CounterexampleStatus,
    Domain,
    RefuteLoopResult,
    RefuteLoopStatus,
    RefuterStrategy,
)

logger = logging.getLogger(__name__)

_BOOTSTRAP_N = 5
_BOOTSTRAP_ITERATIONS = 1000


def _bootstrap_refute_ci(
    paired: "list[tuple[bool, bool]]",
    *,
    n_bootstrap: int = _BOOTSTRAP_ITERATIONS,
    ci_level: float = 0.95,
    seed: int = 42,
) -> "dict[str, tuple[float, float] | None]":
    """Joint percentile bootstrap CI for precision, recall, and F1.

    Args:
        paired: List of ``(is_refuted, is_truly_false)`` pairs.

    Returns:
        Dict with keys ``"precision"``, ``"recall"``, ``"f1"`` each mapping to
        ``(lower, upper)`` or ``None`` when there are fewer than ``_BOOTSTRAP_N``
        samples.
    """
    n = len(paired)
    null: dict[str, tuple[float, float] | None] = {
        "precision": None, "recall": None, "f1": None
    }
    if n < _BOOTSTRAP_N:
        return null
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.array(paired, dtype=np.float64)  # (n, 2): col0=is_refuted, col1=is_truly_false
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = arr[idx]  # (n_bootstrap, n, 2)
    boot_refuted = boot[:, :, 0]
    boot_truly_false = boot[:, :, 1]
    boot_tp = (boot_refuted * boot_truly_false).sum(axis=1)
    boot_n_refuted = boot_refuted.sum(axis=1)
    boot_n_truly_false = boot_truly_false.sum(axis=1)
    boot_precision = boot_tp / np.maximum(boot_n_refuted, 1)
    boot_recall = boot_tp / np.maximum(boot_n_truly_false, 1)
    denom = boot_precision + boot_recall
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_f1 = np.where(denom > 0, 2 * boot_precision * boot_recall / denom, 0.0)
    alpha = 1.0 - ci_level
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    return {
        "precision": (
            float(np.percentile(boot_precision, lo_pct)),
            float(np.percentile(boot_precision, hi_pct)),
        ),
        "recall": (
            float(np.percentile(boot_recall, lo_pct)),
            float(np.percentile(boot_recall, hi_pct)),
        ),
        "f1": (
            float(np.percentile(boot_f1, lo_pct)),
            float(np.percentile(boot_f1, hi_pct)),
        ),
    }


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RefuteMetrics:
    """
    All evaluation metrics for a single REFUTE system run.

    Attributes:
        n_total: Total conjectures evaluated.
        n_refuted: Conjectures for which a confirmed counterexample was found.
        n_survived: Conjectures that survived all refutation rounds.
        n_refined: Conjectures that were refined after counterexample found.
        precision: Fraction of REFUTED predictions that are true positives.
        recall: Fraction of ground-truth-false conjectures that were refuted.
        false_positive_rate: Fraction of truly-true conjectures incorrectly refuted.
        f1: Harmonic mean of precision and recall.
        mean_rounds: Average R-Agent rounds consumed when a counterexample was found.
        strategy_breakdown: Per-strategy statistics dict with keys
            ``{"attempts": N, "successes": M, "rate": R}``.
        domain_breakdown: Per-domain statistics with keys
            ``{"n_total": N, "n_refuted": M, "precision": P, "recall": R}``.
        tier_breakdown: Per-benchmark-tier statistics (same schema as domain_breakdown).
        refinement_rate: Fraction of refuted conjectures that were also refined.
        mean_wall_seconds_per_conjecture: Mean wall-clock seconds per conjecture.
    """

    n_total: int
    n_refuted: int
    n_survived: int
    n_refined: int

    precision: float
    recall: float
    false_positive_rate: float
    f1: float

    mean_rounds: float
    strategy_breakdown: dict[str, dict]
    domain_breakdown: dict[str, dict]
    tier_breakdown: dict[str, dict]

    refinement_rate: float

    mean_wall_seconds_per_conjecture: float = 0.0

    precision_ci_lower: Optional[float] = None
    precision_ci_upper: Optional[float] = None
    recall_ci_lower: Optional[float] = None
    recall_ci_upper: Optional[float] = None
    f1_ci_lower: Optional[float] = None
    f1_ci_upper: Optional[float] = None


@dataclass
class AblationResults:
    """
    Metrics for all ablation study variants reported in Table 2 of the paper.

    Attributes:
        full_system: Metrics from the complete REFUTE system.
        no_strategist: Metrics when strategy is always BOUNDARY (no S-Agent).
        no_refinement: Metrics when the C-Agent refinement step is disabled.
        no_llm_r_agent: Metrics when the R-Agent uses only numerical checking
            (no LLM generation of candidate counterexamples).
    """

    full_system: RefuteMetrics
    no_strategist: RefuteMetrics
    no_refinement: RefuteMetrics
    no_llm_r_agent: RefuteMetrics


# ---------------------------------------------------------------------------
# RefuteEvaluator
# ---------------------------------------------------------------------------


class RefuteEvaluator:
    """
    Computes and reports REFUTE-specific evaluation metrics.

    All heavy computation lives in :meth:`evaluate`.  The result object can be
    rendered, persisted, or converted to LaTeX with the helper methods.

    Design notes:

    - Ground truth is read from ``BenchmarkEntry.ground_truth_status``.
    - Positive class (should be refuted): ``ground_truth_status == "false"``.
    - True positive: REFUTED outcome AND ``ground_truth_status == "false"``.
    - False positive: REFUTED outcome AND ``ground_truth_status != "false"``.
    - False negative: SURVIVED/BUDGET_EXHAUSTED AND ``ground_truth_status == "false"``.
    """

    # ------------------------------------------------------------------
    # Primary evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        loop_results: list[RefuteLoopResult],
        benchmark_entries: list[BenchmarkEntry],
    ) -> RefuteMetrics:
        """
        Compute the full :class:`RefuteMetrics` from a completed REFUTE run.

        Args:
            loop_results: One :class:`~conjlean.schemas.RefuteLoopResult` per
                conjecture processed by the REFUTE loop.
            benchmark_entries: Corresponding benchmark entries providing
                ground-truth labels. Must be in the same order as
                ``loop_results`` or share matching conjecture IDs.

        Returns:
            A fully populated :class:`RefuteMetrics` instance.

        Raises:
            ValueError: If either list is empty or the lengths differ.
        """
        if not loop_results:
            raise ValueError("loop_results must not be empty.")
        if not benchmark_entries:
            raise ValueError("benchmark_entries must not be empty.")
        if len(loop_results) != len(benchmark_entries):
            raise ValueError(
                f"loop_results length ({len(loop_results)}) must match "
                f"benchmark_entries length ({len(benchmark_entries)})."
            )

        n_total = len(loop_results)

        # Build a lookup from conjecture_id -> BenchmarkEntry for O(1) access
        entry_by_id: dict[str, BenchmarkEntry] = {
            e.id: e for e in benchmark_entries
        }

        # Pair results with their benchmark entries via conjecture ID
        paired: list[tuple[RefuteLoopResult, BenchmarkEntry]] = []
        for loop_result, entry in zip(loop_results, benchmark_entries):
            # Prefer positional pairing; fall back to ID-based when IDs mismatch
            if loop_result.original_conjecture.id == entry.conjecture.id:
                paired.append((loop_result, entry))
            elif loop_result.original_conjecture.id in entry_by_id:
                paired.append(
                    (loop_result, entry_by_id[loop_result.original_conjecture.id])
                )
            else:
                logger.warning(
                    "No matching BenchmarkEntry for conjecture %s — using positional pairing.",
                    loop_result.original_conjecture.id,
                )
                paired.append((loop_result, entry))

        # ── Core counts ──────────────────────────────────────────────────────
        n_refuted = sum(
            1 for lr, _ in paired
            if lr.status in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
        )
        n_survived = n_total - n_refuted
        n_refined = sum(1 for lr, _ in paired if lr.refinements)

        # ── Precision / recall / FPR ─────────────────────────────────────────
        true_positives = sum(
            1 for lr, be in paired
            if lr.status in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
            and be.ground_truth_status == "false"
        )
        false_positives = sum(
            1 for lr, be in paired
            if lr.status in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
            and be.ground_truth_status != "false"
        )
        false_negatives = sum(
            1 for lr, be in paired
            if lr.status not in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
            and be.ground_truth_status == "false"
        )
        n_truly_false = true_positives + false_negatives
        n_truly_not_false = false_positives + sum(
            1 for lr, be in paired
            if lr.status not in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
            and be.ground_truth_status != "false"
        )

        precision = true_positives / max(n_refuted, 1)
        recall = true_positives / max(n_truly_false, 1)
        false_positive_rate = false_positives / max(n_truly_not_false, 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # ── Mean rounds ──────────────────────────────────────────────────────
        rounds_when_found: list[int] = [
            lr.total_rounds
            for lr, _ in paired
            if lr.status in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
            and lr.total_rounds > 0
        ]
        mean_rounds = (
            sum(rounds_when_found) / len(rounds_when_found)
            if rounds_when_found else 0.0
        )

        # ── Breakdown metrics ─────────────────────────────────────────────────
        strategy_breakdown = self.compute_strategy_breakdown(loop_results)
        domain_breakdown = self.compute_domain_breakdown(loop_results, benchmark_entries)
        tier_breakdown = self.compute_tier_breakdown(loop_results, benchmark_entries)

        # ── Refinement rate ───────────────────────────────────────────────────
        refinement_rate = n_refined / max(n_refuted, 1)

        # ── Bootstrap confidence intervals ────────────────────────────────────
        paired_outcomes = [
            (
                lr.status in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED),
                be.ground_truth_status == "false",
            )
            for lr, be in paired
        ]
        cis = _bootstrap_refute_ci(paired_outcomes)

        metrics = RefuteMetrics(
            n_total=n_total,
            n_refuted=n_refuted,
            n_survived=n_survived,
            n_refined=n_refined,
            precision=round(precision, 6),
            recall=round(recall, 6),
            false_positive_rate=round(false_positive_rate, 6),
            f1=round(f1, 6),
            mean_rounds=round(mean_rounds, 4),
            strategy_breakdown=strategy_breakdown,
            domain_breakdown=domain_breakdown,
            tier_breakdown=tier_breakdown,
            refinement_rate=round(refinement_rate, 6),
            mean_wall_seconds_per_conjecture=0.0,
            precision_ci_lower=cis["precision"][0] if cis["precision"] else None,
            precision_ci_upper=cis["precision"][1] if cis["precision"] else None,
            recall_ci_lower=cis["recall"][0] if cis["recall"] else None,
            recall_ci_upper=cis["recall"][1] if cis["recall"] else None,
            f1_ci_lower=cis["f1"][0] if cis["f1"] else None,
            f1_ci_upper=cis["f1"][1] if cis["f1"] else None,
        )

        logger.info(
            "RefuteEvaluator.evaluate | n_total=%d | n_refuted=%d | "
            "precision=%.4f | recall=%.4f | f1=%.4f",
            n_total,
            n_refuted,
            precision,
            recall,
            f1,
        )
        return metrics

    # ------------------------------------------------------------------
    # Breakdown computations
    # ------------------------------------------------------------------

    def compute_strategy_breakdown(
        self,
        loop_results: list[RefuteLoopResult],
    ) -> dict[str, dict]:
        """
        Compute per-strategy attempt and success counts across all loop results.

        For each RefuterStrategy, reports total attempts, successful refutations,
        and the empirical success rate.

        Args:
            loop_results: All :class:`~conjlean.schemas.RefuteLoopResult` objects
                from a completed run.

        Returns:
            Dict keyed by strategy name with nested dicts containing
            ``{"attempts": int, "successes": int, "rate": float}``.
        """
        counts: dict[str, dict[str, int]] = {
            s.value: {"attempts": 0, "successes": 0}
            for s in RefuterStrategy
        }

        for loop_result in loop_results:
            for refuter_result in loop_result.refuter_results:
                strategy = refuter_result.strategy_used
                if strategy is None:
                    continue
                key = strategy.value
                counts[key]["attempts"] += 1
                if refuter_result.best_counterexample is not None:
                    counts[key]["successes"] += 1

            # Also count from strategy_scores on individual refuter results
            for refuter_result in loop_result.refuter_results:
                for strat_key, score in refuter_result.strategy_scores.items():
                    if strat_key in counts and isinstance(score, int):
                        pass  # strategy_scores track cumulative; already counted above

        breakdown: dict[str, dict] = {}
        for strat_key, vals in counts.items():
            attempts = vals["attempts"]
            successes = vals["successes"]
            breakdown[strat_key] = {
                "attempts": attempts,
                "successes": successes,
                "rate": round(successes / max(attempts, 1), 6),
            }

        return breakdown

    def compute_domain_breakdown(
        self,
        loop_results: list[RefuteLoopResult],
        benchmark_entries: list[BenchmarkEntry],
    ) -> dict[str, dict]:
        """
        Compute per-domain precision, recall, and refutation counts.

        Args:
            loop_results: Ordered list of loop results.
            benchmark_entries: Corresponding benchmark entries (same order).

        Returns:
            Dict keyed by domain value with nested dicts containing
            ``{"n_total": int, "n_refuted": int, "n_truly_false": int,
            "true_positives": int, "precision": float, "recall": float}``.

        Raises:
            ValueError: If the two lists have different lengths.
        """
        if len(loop_results) != len(benchmark_entries):
            raise ValueError(
                "loop_results and benchmark_entries must have the same length."
            )

        domain_data: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "n_total": 0,
                "n_refuted": 0,
                "n_truly_false": 0,
                "true_positives": 0,
            }
        )

        for lr, be in zip(loop_results, benchmark_entries):
            domain = be.conjecture.domain.value
            domain_data[domain]["n_total"] += 1

            is_refuted = lr.status in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
            is_truly_false = be.ground_truth_status == "false"

            if is_refuted:
                domain_data[domain]["n_refuted"] += 1
            if is_truly_false:
                domain_data[domain]["n_truly_false"] += 1
            if is_refuted and is_truly_false:
                domain_data[domain]["true_positives"] += 1

        breakdown: dict[str, dict] = {}
        for domain, vals in domain_data.items():
            n_refuted = vals["n_refuted"]
            true_positives = vals["true_positives"]
            n_truly_false = vals["n_truly_false"]
            breakdown[domain] = {
                "n_total": vals["n_total"],
                "n_refuted": n_refuted,
                "n_truly_false": n_truly_false,
                "true_positives": true_positives,
                "precision": round(true_positives / max(n_refuted, 1), 6),
                "recall": round(true_positives / max(n_truly_false, 1), 6),
            }

        return breakdown

    def compute_tier_breakdown(
        self,
        loop_results: list[RefuteLoopResult],
        benchmark_entries: list[BenchmarkEntry],
    ) -> dict[str, dict]:
        """
        Compute per-benchmark-tier precision, recall, and refutation counts.

        Args:
            loop_results: Ordered list of loop results.
            benchmark_entries: Corresponding benchmark entries (same order).

        Returns:
            Dict keyed by tier value (e.g. ``"tier1_synthetic"``) with nested
            dicts containing the same schema as :meth:`compute_domain_breakdown`.

        Raises:
            ValueError: If the two lists have different lengths.
        """
        if len(loop_results) != len(benchmark_entries):
            raise ValueError(
                "loop_results and benchmark_entries must have the same length."
            )

        tier_data: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "n_total": 0,
                "n_refuted": 0,
                "n_truly_false": 0,
                "true_positives": 0,
            }
        )

        for lr, be in zip(loop_results, benchmark_entries):
            tier = be.tier.value
            tier_data[tier]["n_total"] += 1

            is_refuted = lr.status in (RefuteLoopStatus.REFUTED, RefuteLoopStatus.REFINED)
            is_truly_false = be.ground_truth_status == "false"

            if is_refuted:
                tier_data[tier]["n_refuted"] += 1
            if is_truly_false:
                tier_data[tier]["n_truly_false"] += 1
            if is_refuted and is_truly_false:
                tier_data[tier]["true_positives"] += 1

        breakdown: dict[str, dict] = {}
        for tier, vals in tier_data.items():
            n_refuted = vals["n_refuted"]
            true_positives = vals["true_positives"]
            n_truly_false = vals["n_truly_false"]
            breakdown[tier] = {
                "n_total": vals["n_total"],
                "n_refuted": n_refuted,
                "n_truly_false": n_truly_false,
                "true_positives": true_positives,
                "precision": round(true_positives / max(n_refuted, 1), 6),
                "recall": round(true_positives / max(n_truly_false, 1), 6),
            }

        return breakdown

    def compute_refinement_quality(
        self,
        loop_results: list[RefuteLoopResult],
        benchmark_entries: list[BenchmarkEntry],
    ) -> dict[str, float]:
        """
        Heuristically assess whether refined conjectures are "more correct".

        A refinement is considered successful when the refined conjecture's
        statement differs from the original AND the loop did not produce a
        further counterexample against the refined version (i.e., the loop
        terminated with status REFINED rather than cycling back to REFUTED).

        Args:
            loop_results: Ordered list of loop results.
            benchmark_entries: Corresponding benchmark entries.

        Returns:
            Dict with keys:
                - ``"n_refined_total"``: total refinements attempted.
                - ``"n_refined_changed"``: refinements where statement changed.
                - ``"changed_rate"``: fraction of refinements that changed the statement.
                - ``"refinement_survival_rate"``: fraction of refined conjectures
                  where the final conjecture survived further refutation.
        """
        if len(loop_results) != len(benchmark_entries):
            raise ValueError(
                "loop_results and benchmark_entries must have the same length."
            )

        n_refined_total = 0
        n_refined_changed = 0
        n_survived_after_refinement = 0

        for lr, _ in zip(loop_results, benchmark_entries):
            if not lr.refinements:
                continue

            for refinement in lr.refinements:
                n_refined_total += 1
                original_stmt = refinement.original.nl_statement.strip()
                refined_stmt = refinement.refined_statement.strip()
                if original_stmt != refined_stmt:
                    n_refined_changed += 1

            # If the loop terminated with REFINED (not REFUTED), refinement improved survivability
            if lr.status == RefuteLoopStatus.REFINED:
                n_survived_after_refinement += 1

        total_refined_loops = sum(1 for lr, _ in zip(loop_results, benchmark_entries) if lr.refinements)

        return {
            "n_refined_total": n_refined_total,
            "n_refined_changed": n_refined_changed,
            "changed_rate": round(n_refined_changed / max(n_refined_total, 1), 6),
            "refinement_survival_rate": round(
                n_survived_after_refinement / max(total_refined_loops, 1), 6
            ),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, metrics: RefuteMetrics) -> None:
        """
        Print a formatted REFUTE evaluation report to stdout.

        Attempts to use :mod:`rich` tables for a polished output; falls back
        to plain ASCII tables if :mod:`rich` is not installed.

        Args:
            metrics: A fully populated :class:`RefuteMetrics` instance.
        """
        try:
            self._print_rich(metrics)
        except ImportError:
            self._print_plain(metrics)

    def save_report(self, metrics: RefuteMetrics, path: Path) -> None:
        """
        Persist the evaluation report as both a JSON file and a Markdown file.

        Creates parent directories as needed.  The ``path`` argument should
        omit the file extension; ``.json`` and ``.md`` are appended automatically.

        Args:
            metrics: A fully populated :class:`RefuteMetrics` instance.
            path: Base path for output files (no extension).
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        json_path = path.with_suffix(".json")
        md_path = path.with_suffix(".md")

        report_dict = self._metrics_to_dict(metrics)
        json_path.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(self._metrics_to_markdown(metrics), encoding="utf-8")

        logger.info(
            "RefuteEvaluator: report saved to %s and %s", json_path, md_path
        )

    def to_latex_table(self, ablation: Optional[AblationResults] = None) -> str:
        """
        Generate a LaTeX table string for inclusion in the ICML paper.

        When ``ablation`` is provided, renders the full ablation comparison
        table (Table 2).  When ``ablation`` is ``None``, generates a minimal
        placeholder table.

        Args:
            ablation: Optional :class:`AblationResults` from an ablation run.
                If ``None``, returns an empty scaffold.

        Returns:
            A complete LaTeX tabular environment string.
        """
        if ablation is None:
            return (
                r"\begin{table}[h]" + "\n"
                r"\centering" + "\n"
                r"\begin{tabular}{lrrr}" + "\n"
                r"\hline" + "\n"
                r"\textbf{Method} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\" + "\n"
                r"\hline" + "\n"
                r"REFUTE (Full) & -- & -- & -- \\" + "\n"
                r"\hline" + "\n"
                r"\end{tabular}" + "\n"
                r"\caption{REFUTE system performance on 3-tier benchmark.}" + "\n"
                r"\label{tab:main_results}" + "\n"
                r"\end{table}"
            )

        rows = [
            ("REFUTE (Full)", ablation.full_system),
            ("No Strategist", ablation.no_strategist),
            ("No Refinement", ablation.no_refinement),
            ("No LLM", ablation.no_llm_r_agent),
        ]

        body_lines: list[str] = []
        for method_name, m in rows:
            body_lines.append(
                f"{method_name} & {m.precision:.2f} & {m.recall:.2f} & {m.f1:.2f} \\\\"
            )

        body = "\n".join(body_lines)

        return (
            r"\begin{table}[h]" + "\n"
            r"\centering" + "\n"
            r"\begin{tabular}{lrrr}" + "\n"
            r"\hline" + "\n"
            r"\textbf{Method} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\" + "\n"
            r"\hline" + "\n"
            + body + "\n"
            r"\hline" + "\n"
            r"\end{tabular}" + "\n"
            r"\caption{REFUTE system performance on 3-tier benchmark.}" + "\n"
            r"\label{tab:main_results}" + "\n"
            r"\end{table}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_to_dict(metrics: RefuteMetrics) -> dict:
        """
        Convert a :class:`RefuteMetrics` to a JSON-serialisable dict.

        Args:
            metrics: The metrics object to convert.

        Returns:
            A flat/nested dictionary suitable for ``json.dumps``.
        """
        d: dict = {
            "n_total": metrics.n_total,
            "n_refuted": metrics.n_refuted,
            "n_survived": metrics.n_survived,
            "n_refined": metrics.n_refined,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "false_positive_rate": metrics.false_positive_rate,
            "f1": metrics.f1,
            "mean_rounds": metrics.mean_rounds,
            "refinement_rate": metrics.refinement_rate,
            "mean_wall_seconds_per_conjecture": metrics.mean_wall_seconds_per_conjecture,
            "strategy_breakdown": metrics.strategy_breakdown,
            "domain_breakdown": metrics.domain_breakdown,
            "tier_breakdown": metrics.tier_breakdown,
        }
        if metrics.precision_ci_lower is not None:
            d["precision_ci"] = [
                round(metrics.precision_ci_lower, 6),
                round(metrics.precision_ci_upper, 6),
            ]
            d["recall_ci"] = [
                round(metrics.recall_ci_lower, 6),
                round(metrics.recall_ci_upper, 6),
            ]
            d["f1_ci"] = [
                round(metrics.f1_ci_lower, 6),
                round(metrics.f1_ci_upper, 6),
            ]
        return d

    @staticmethod
    def _metrics_to_markdown(metrics: RefuteMetrics) -> str:
        """
        Render a :class:`RefuteMetrics` as a Markdown document.

        Args:
            metrics: The metrics object to render.

        Returns:
            A multi-section Markdown string.
        """
        lines: list[str] = ["# REFUTE Evaluation Report\n"]

        # ── Primary metrics ──────────────────────────────────────────────────
        lines.append("## Primary Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Total conjectures | {metrics.n_total} |")
        lines.append(f"| Refuted | {metrics.n_refuted} |")
        lines.append(f"| Survived | {metrics.n_survived} |")
        lines.append(f"| Refined | {metrics.n_refined} |")
        if metrics.precision_ci_lower is not None:
            lines.append(
                f"| Precision | {metrics.precision:.4f} "
                f"[{metrics.precision_ci_lower:.4f}, {metrics.precision_ci_upper:.4f}] |"
            )
            lines.append(
                f"| Recall | {metrics.recall:.4f} "
                f"[{metrics.recall_ci_lower:.4f}, {metrics.recall_ci_upper:.4f}] |"
            )
            lines.append(
                f"| F1 | {metrics.f1:.4f} "
                f"[{metrics.f1_ci_lower:.4f}, {metrics.f1_ci_upper:.4f}] |"
            )
        else:
            lines.append(f"| Precision | {metrics.precision:.4f} |")
            lines.append(f"| Recall | {metrics.recall:.4f} |")
            lines.append(f"| F1 | {metrics.f1:.4f} |")
        lines.append(f"| False Positive Rate | {metrics.false_positive_rate:.4f} |")
        lines.append(f"| Mean rounds (when found) | {metrics.mean_rounds:.2f} |")
        lines.append(f"| Refinement rate | {metrics.refinement_rate:.4f} |")
        lines.append("")

        # ── Strategy breakdown ───────────────────────────────────────────────
        if metrics.strategy_breakdown:
            lines.append("## Strategy Breakdown\n")
            lines.append("| Strategy | Attempts | Successes | Rate |")
            lines.append("|---|---|---|---|")
            for strat, vals in sorted(metrics.strategy_breakdown.items()):
                lines.append(
                    f"| {strat} | {vals['attempts']} | {vals['successes']} "
                    f"| {vals['rate']:.4f} |"
                )
            lines.append("")

        # ── Domain breakdown ─────────────────────────────────────────────────
        if metrics.domain_breakdown:
            lines.append("## Domain Breakdown\n")
            lines.append("| Domain | Total | Refuted | Truly False | TP | Precision | Recall |")
            lines.append("|---|---|---|---|---|---|---|")
            for domain, vals in sorted(metrics.domain_breakdown.items()):
                lines.append(
                    f"| {domain} | {vals['n_total']} | {vals['n_refuted']} "
                    f"| {vals['n_truly_false']} | {vals['true_positives']} "
                    f"| {vals['precision']:.4f} | {vals['recall']:.4f} |"
                )
            lines.append("")

        # ── Tier breakdown ───────────────────────────────────────────────────
        if metrics.tier_breakdown:
            lines.append("## Tier Breakdown\n")
            lines.append("| Tier | Total | Refuted | Truly False | TP | Precision | Recall |")
            lines.append("|---|---|---|---|---|---|---|")
            for tier, vals in sorted(metrics.tier_breakdown.items()):
                lines.append(
                    f"| {tier} | {vals['n_total']} | {vals['n_refuted']} "
                    f"| {vals['n_truly_false']} | {vals['true_positives']} "
                    f"| {vals['precision']:.4f} | {vals['recall']:.4f} |"
                )
            lines.append("")

        return "\n".join(lines)

    def _print_rich(self, metrics: RefuteMetrics) -> None:
        """
        Render the metrics using :mod:`rich` Console and Table objects.

        Args:
            metrics: The metrics object to render.

        Raises:
            ImportError: If :mod:`rich` is not installed.
        """
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Primary metrics table
        primary = Table(
            title="REFUTE Primary Metrics",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        primary.add_column("Metric", style="bold")
        primary.add_column("Value", justify="right")
        def _ci_suffix(lo: Optional[float], hi: Optional[float]) -> str:
            return f" [{lo:.4f}, {hi:.4f}]" if lo is not None else ""

        for label, value in [
            ("Total conjectures", str(metrics.n_total)),
            ("Refuted", str(metrics.n_refuted)),
            ("Survived", str(metrics.n_survived)),
            ("Refined", str(metrics.n_refined)),
            (
                "Precision",
                f"{metrics.precision:.4f}"
                + _ci_suffix(metrics.precision_ci_lower, metrics.precision_ci_upper),
            ),
            (
                "Recall",
                f"{metrics.recall:.4f}"
                + _ci_suffix(metrics.recall_ci_lower, metrics.recall_ci_upper),
            ),
            (
                "F1",
                f"{metrics.f1:.4f}"
                + _ci_suffix(metrics.f1_ci_lower, metrics.f1_ci_upper),
            ),
            ("False Positive Rate", f"{metrics.false_positive_rate:.4f}"),
            ("Mean rounds (when found)", f"{metrics.mean_rounds:.2f}"),
            ("Refinement rate", f"{metrics.refinement_rate:.4f}"),
        ]:
            primary.add_row(label, value)
        console.print(primary)

        # Strategy breakdown table
        if any(v["attempts"] > 0 for v in metrics.strategy_breakdown.values()):
            strat_table = Table(
                title="Strategy Breakdown",
                box=box.SIMPLE,
                header_style="bold magenta",
            )
            strat_table.add_column("Strategy")
            strat_table.add_column("Attempts", justify="right")
            strat_table.add_column("Successes", justify="right")
            strat_table.add_column("Rate", justify="right")
            for strat, vals in sorted(metrics.strategy_breakdown.items()):
                strat_table.add_row(
                    strat,
                    str(vals["attempts"]),
                    str(vals["successes"]),
                    f"{vals['rate']:.4f}",
                )
            console.print(strat_table)

        # Domain breakdown table
        if metrics.domain_breakdown:
            domain_table = Table(
                title="Domain Breakdown",
                box=box.SIMPLE,
                header_style="bold green",
            )
            domain_table.add_column("Domain")
            domain_table.add_column("Total", justify="right")
            domain_table.add_column("Refuted", justify="right")
            domain_table.add_column("Precision", justify="right")
            domain_table.add_column("Recall", justify="right")
            for domain, vals in sorted(metrics.domain_breakdown.items()):
                domain_table.add_row(
                    domain,
                    str(vals["n_total"]),
                    str(vals["n_refuted"]),
                    f"{vals['precision']:.4f}",
                    f"{vals['recall']:.4f}",
                )
            console.print(domain_table)

        # Tier breakdown table
        if metrics.tier_breakdown:
            tier_table = Table(
                title="Tier Breakdown",
                box=box.SIMPLE,
                header_style="bold yellow",
            )
            tier_table.add_column("Tier")
            tier_table.add_column("Total", justify="right")
            tier_table.add_column("Refuted", justify="right")
            tier_table.add_column("Precision", justify="right")
            tier_table.add_column("Recall", justify="right")
            for tier, vals in sorted(metrics.tier_breakdown.items()):
                tier_table.add_row(
                    tier,
                    str(vals["n_total"]),
                    str(vals["n_refuted"]),
                    f"{vals['precision']:.4f}",
                    f"{vals['recall']:.4f}",
                )
            console.print(tier_table)

    def _print_plain(self, metrics: RefuteMetrics) -> None:
        """
        Render the metrics as plain-text ASCII tables (no dependencies).

        Args:
            metrics: The metrics object to render.
        """
        def _row(cols: list[str], widths: list[int]) -> str:
            return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"

        def _sep(widths: list[int]) -> str:
            return "+-" + "-+-".join("-" * w for w in widths) + "-+"

        print("\n=== REFUTE Evaluation Report ===\n")
        print(f"  Total: {metrics.n_total}  |  Refuted: {metrics.n_refuted}  "
              f"|  Survived: {metrics.n_survived}  |  Refined: {metrics.n_refined}")
        if metrics.precision_ci_lower is not None:
            print(
                f"  Precision: {metrics.precision:.4f} "
                f"[{metrics.precision_ci_lower:.4f}, {metrics.precision_ci_upper:.4f}]  |  "
                f"Recall: {metrics.recall:.4f} "
                f"[{metrics.recall_ci_lower:.4f}, {metrics.recall_ci_upper:.4f}]  |  "
                f"F1: {metrics.f1:.4f} "
                f"[{metrics.f1_ci_lower:.4f}, {metrics.f1_ci_upper:.4f}]  |  "
                f"FPR: {metrics.false_positive_rate:.4f}"
            )
        else:
            print(
                f"  Precision: {metrics.precision:.4f}  |  Recall: {metrics.recall:.4f}  "
                f"|  F1: {metrics.f1:.4f}  |  FPR: {metrics.false_positive_rate:.4f}"
            )
        print(f"  Mean rounds (when found): {metrics.mean_rounds:.2f}  "
              f"|  Refinement rate: {metrics.refinement_rate:.4f}")

        if any(v["attempts"] > 0 for v in metrics.strategy_breakdown.values()):
            print("\n  Strategy Breakdown:")
            headers = ["Strategy", "Attempts", "Successes", "Rate"]
            rows_data = [
                [s, str(v["attempts"]), str(v["successes"]), f"{v['rate']:.4f}"]
                for s, v in sorted(metrics.strategy_breakdown.items())
            ]
            widths = [
                max(len(headers[i]), max(len(r[i]) for r in rows_data))
                for i in range(len(headers))
            ]
            print("  " + _sep(widths))
            print("  " + _row(headers, widths))
            print("  " + _sep(widths))
            for row in rows_data:
                print("  " + _row(row, widths))
            print("  " + _sep(widths))

        if metrics.domain_breakdown:
            print("\n  Domain Breakdown:")
            for domain, vals in sorted(metrics.domain_breakdown.items()):
                print(
                    f"    {domain}: total={vals['n_total']}  refuted={vals['n_refuted']}  "
                    f"precision={vals['precision']:.4f}  recall={vals['recall']:.4f}"
                )

        if metrics.tier_breakdown:
            print("\n  Tier Breakdown:")
            for tier, vals in sorted(metrics.tier_breakdown.items()):
                print(
                    f"    {tier}: total={vals['n_total']}  refuted={vals['n_refuted']}  "
                    f"precision={vals['precision']:.4f}  recall={vals['recall']:.4f}"
                )
        print()
