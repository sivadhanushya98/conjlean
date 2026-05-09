"""
Evaluation and reporting for the ConjLean pipeline.

The :class:`Evaluator` ingests a list of
:class:`~conjlean.schemas.PipelineResult` objects and produces a
structured :class:`EvaluationReport` covering every pipeline stage.

Metrics computed:

- Per-stage success rates (:class:`StageMetrics`)
- Layer-by-layer proof closure counts
- Domain-level breakdown of outcomes
- Formalization error taxonomy (classified by Lean error pattern)
- Mean formalization retries and proof durations
- Markdown and JSON export methods
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from conjlean.schemas import (
    FilterStatus,
    FormalizationStatus,
    PipelineResult,
    PipelineStatus,
    ProofLayer,
    ProofStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

_MIN_BOOTSTRAP_N = 5
_BOOTSTRAP_ITERATIONS = 1000


def bootstrap_ci(
    outcomes: list[bool],
    *,
    n_bootstrap: int = _BOOTSTRAP_ITERATIONS,
    ci_level: float = 0.95,
    seed: int = 42,
) -> "tuple[float, float] | None":
    """Percentile bootstrap CI for a Bernoulli rate.

    Returns ``(lower, upper)`` or ``None`` when ``len(outcomes) < _MIN_BOOTSTRAP_N``.
    """
    n = len(outcomes)
    if n < _MIN_BOOTSTRAP_N:
        return None
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.array(outcomes, dtype=np.float64)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    """
    Aggregate success statistics for a single pipeline stage.

    Attributes:
        total: Total items entering the stage.
        success: Items that passed / succeeded at this stage.
        rate: ``success / total`` as a float in ``[0.0, 1.0]``.
        breakdown: Status-keyed counts of all outcomes at this stage.
    """

    total: int
    success: int
    rate: float
    breakdown: dict[str, int] = field(default_factory=dict)
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


@dataclass
class EvaluationReport:
    """
    Full evaluation report for a completed ConjLean pipeline run.

    Attributes:
        generation: Metrics for the conjecture generation stage.
        filtering: Metrics for the symbolic / numerical filtering stage.
        formalization: Metrics for the Lean 4 formalization stage.
        proof_search: Metrics for the layered proof search stage.
        end_to_end: End-to-end metrics (proved / total input conjectures).
        layer_breakdown: Map of proof-layer name → count of proofs closed.
        domain_breakdown: Nested map of ``domain → {status: count}``.
        error_taxonomy: Map of Lean error category → occurrence count.
        mean_formalization_retries: Average LLM retries consumed per
            formalized conjecture.
        mean_proof_duration_seconds: Average wall-clock seconds spent on
            proof search per conjecture (proved or open).
    """

    generation: StageMetrics
    filtering: StageMetrics
    formalization: StageMetrics
    proof_search: StageMetrics
    end_to_end: StageMetrics

    layer_breakdown: dict[str, int] = field(default_factory=dict)
    domain_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    error_taxonomy: dict[str, int] = field(default_factory=dict)

    mean_formalization_retries: float = 0.0
    mean_proof_duration_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """
        Render the report as a Markdown document with ASCII tables.

        Returns:
            Multi-section Markdown string suitable for inclusion in a
            research paper appendix or GitHub wiki page.
        """
        lines: list[str] = ["# ConjLean Evaluation Report\n"]

        # ── Pipeline Stage Summary ──────────────────────────────────────
        stage_pairs = [
            ("Generation", self.generation),
            ("Filtering (surviving)", self.filtering),
            ("Formalization (typechecks)", self.formalization),
            ("Proof Search (proved)", self.proof_search),
            ("End-to-End (proved)", self.end_to_end),
        ]
        has_ci = any(m.ci_lower is not None for _, m in stage_pairs)
        lines.append("## Pipeline Stage Summary\n")
        if has_ci:
            lines.append("| Stage | Total | Success | Rate | 95% CI |")
            lines.append("|---|---|---|---|---|")
            for stage_name, metrics in stage_pairs:
                ci_str = (
                    f"[{metrics.ci_lower:.1%}, {metrics.ci_upper:.1%}]"
                    if metrics.ci_lower is not None
                    else "—"
                )
                lines.append(
                    f"| {stage_name} | {metrics.total} | {metrics.success} "
                    f"| {metrics.rate:.1%} | {ci_str} |"
                )
        else:
            lines.append("| Stage | Total | Success | Rate |")
            lines.append("|---|---|---|---|")
            for stage_name, metrics in stage_pairs:
                lines.append(
                    f"| {stage_name} | {metrics.total} | {metrics.success} "
                    f"| {metrics.rate:.1%} |"
                )
        lines.append("")

        # ── Layer Breakdown ─────────────────────────────────────────────
        if self.layer_breakdown:
            lines.append("## Proof Layer Breakdown\n")
            lines.append("| Layer | Proofs Closed |")
            lines.append("|---|---|")
            for layer, count in sorted(self.layer_breakdown.items()):
                lines.append(f"| {layer} | {count} |")
            lines.append("")

        # ── Domain Breakdown ────────────────────────────────────────────
        if self.domain_breakdown:
            lines.append("## Domain Breakdown\n")
            all_statuses = sorted(
                {s for counts in self.domain_breakdown.values() for s in counts}
            )
            header = "| Domain | " + " | ".join(all_statuses) + " |"
            sep = "|---|" + "---|" * len(all_statuses)
            lines.append(header)
            lines.append(sep)
            for domain, counts in sorted(self.domain_breakdown.items()):
                row_vals = " | ".join(str(counts.get(s, 0)) for s in all_statuses)
                lines.append(f"| {domain} | {row_vals} |")
            lines.append("")

        # ── Error Taxonomy ──────────────────────────────────────────────
        if self.error_taxonomy:
            lines.append("## Formalization Error Taxonomy\n")
            lines.append("| Error Type | Count |")
            lines.append("|---|---|")
            for err_type, count in sorted(
                self.error_taxonomy.items(), key=lambda x: -x[1]
            ):
                lines.append(f"| {err_type} | {count} |")
            lines.append("")

        # ── Timing ──────────────────────────────────────────────────────
        lines.append("## Timing\n")
        lines.append(
            f"- Mean formalization retries: **{self.mean_formalization_retries:.2f}**"
        )
        lines.append(
            f"- Mean proof search duration: **{self.mean_proof_duration_seconds:.2f}s**"
        )
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialise the report to a plain nested dictionary.

        All :class:`~conjlean.schemas.StageMetrics` objects are inlined as
        nested dicts so the result is directly JSON-serialisable.

        Returns:
            Dictionary representation of the full report.
        """
        def _metrics_to_dict(m: StageMetrics) -> dict:
            d: dict = {
                "total": m.total,
                "success": m.success,
                "rate": round(m.rate, 6),
                "breakdown": m.breakdown,
            }
            if m.ci_lower is not None:
                d["ci_lower"] = round(m.ci_lower, 6)
                d["ci_upper"] = round(m.ci_upper, 6)
            return d

        return {
            "generation": _metrics_to_dict(self.generation),
            "filtering": _metrics_to_dict(self.filtering),
            "formalization": _metrics_to_dict(self.formalization),
            "proof_search": _metrics_to_dict(self.proof_search),
            "end_to_end": _metrics_to_dict(self.end_to_end),
            "layer_breakdown": self.layer_breakdown,
            "domain_breakdown": self.domain_breakdown,
            "error_taxonomy": self.error_taxonomy,
            "mean_formalization_retries": round(self.mean_formalization_retries, 4),
            "mean_proof_duration_seconds": round(self.mean_proof_duration_seconds, 4),
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """
    Computes and reports evaluation metrics for a completed pipeline run.

    All computation is performed by :meth:`evaluate`; the result object can
    then be printed with :meth:`print_report` or persisted with
    :meth:`save_report`.
    """

    def evaluate(self, results: list[PipelineResult]) -> EvaluationReport:
        """
        Compute the full :class:`EvaluationReport` from pipeline results.

        Args:
            results: One :class:`~conjlean.schemas.PipelineResult` per
                conjecture that entered the pipeline.

        Returns:
            Populated :class:`EvaluationReport`.

        Raises:
            ValueError: If ``results`` is empty.
        """
        if not results:
            raise ValueError("Cannot evaluate an empty results list.")

        total_in = len(results)

        # ── Generation metrics ───────────────────────────────────────────
        generation_metrics = StageMetrics(
            total=total_in,
            success=total_in,
            rate=1.0,
            breakdown={"generated": total_in},
        )

        # ── Filtering metrics ────────────────────────────────────────────
        filter_breakdown: dict[str, int] = defaultdict(int)
        n_surviving = 0
        for r in results:
            if r.filter_result is not None:
                key = r.filter_result.status.value
                filter_breakdown[key] += 1
                if r.filter_result.status is FilterStatus.SURVIVING:
                    n_surviving += 1
            else:
                filter_breakdown["no_filter_result"] += 1

        filter_outcomes = [
            r.filter_result is not None and r.filter_result.status is FilterStatus.SURVIVING
            for r in results
        ]
        filter_ci = bootstrap_ci(filter_outcomes)
        filtering_metrics = StageMetrics(
            total=total_in,
            success=n_surviving,
            rate=n_surviving / total_in,
            breakdown=dict(filter_breakdown),
            ci_lower=filter_ci[0] if filter_ci else None,
            ci_upper=filter_ci[1] if filter_ci else None,
        )

        # ── Formalization metrics ────────────────────────────────────────
        n_formalized_input = sum(
            1
            for r in results
            if r.filter_result is not None
            and r.filter_result.status is FilterStatus.SURVIVING
        )
        formalization_breakdown: dict[str, int] = defaultdict(int)
        n_typechecks = 0
        retry_counts: list[int] = []

        for r in results:
            if r.formalization is not None:
                key = r.formalization.status.value
                formalization_breakdown[key] += 1
                if r.formalization.status is FormalizationStatus.TYPECHECKS:
                    n_typechecks += 1
                retry_counts.append(r.formalization.retries)
            elif (
                r.filter_result is not None
                and r.filter_result.status is FilterStatus.SURVIVING
            ):
                formalization_breakdown["not_attempted"] += 1

        form_outcomes = [
            r.formalization is not None
            and r.formalization.status is FormalizationStatus.TYPECHECKS
            for r in results
            if r.filter_result is not None
            and r.filter_result.status is FilterStatus.SURVIVING
        ]
        form_ci = bootstrap_ci(form_outcomes)
        formalization_metrics = StageMetrics(
            total=max(n_formalized_input, 1),
            success=n_typechecks,
            rate=n_typechecks / max(n_formalized_input, 1),
            breakdown=dict(formalization_breakdown),
            ci_lower=form_ci[0] if form_ci else None,
            ci_upper=form_ci[1] if form_ci else None,
        )
        mean_retries = sum(retry_counts) / max(len(retry_counts), 1)

        # ── Proof search metrics ─────────────────────────────────────────
        proof_breakdown: dict[str, int] = defaultdict(int)
        n_proved = 0
        proof_durations: list[float] = []

        for r in results:
            if r.proof is not None:
                key = r.proof.status.value
                proof_breakdown[key] += 1
                proof_durations.append(r.proof.duration_seconds)
                if r.proof.status is ProofStatus.PROVED:
                    n_proved += 1
            elif r.formalization is not None and r.formalization.status is FormalizationStatus.TYPECHECKS:
                proof_breakdown["not_attempted"] += 1

        n_proof_input = sum(
            1
            for r in results
            if r.formalization is not None
            and r.formalization.status is FormalizationStatus.TYPECHECKS
        )
        proof_outcomes = [
            r.proof is not None and r.proof.status is ProofStatus.PROVED
            for r in results
            if r.formalization is not None
            and r.formalization.status is FormalizationStatus.TYPECHECKS
        ]
        proof_ci = bootstrap_ci(proof_outcomes)
        proof_search_metrics = StageMetrics(
            total=max(n_proof_input, 1),
            success=n_proved,
            rate=n_proved / max(n_proof_input, 1),
            breakdown=dict(proof_breakdown),
            ci_lower=proof_ci[0] if proof_ci else None,
            ci_upper=proof_ci[1] if proof_ci else None,
        )
        mean_duration = sum(proof_durations) / max(len(proof_durations), 1)

        # ── End-to-end metrics ───────────────────────────────────────────
        e2e_breakdown: dict[str, int] = defaultdict(int)
        for r in results:
            e2e_breakdown[r.final_status.value] += 1

        e2e_outcomes = [
            r.proof is not None and r.proof.status is ProofStatus.PROVED
            for r in results
        ]
        e2e_ci = bootstrap_ci(e2e_outcomes)
        end_to_end_metrics = StageMetrics(
            total=total_in,
            success=n_proved,
            rate=n_proved / total_in,
            breakdown=dict(e2e_breakdown),
            ci_lower=e2e_ci[0] if e2e_ci else None,
            ci_upper=e2e_ci[1] if e2e_ci else None,
        )

        # ── Layer / domain / error breakdowns ────────────────────────────
        layer_breakdown = self.compute_layer_breakdown(results)
        domain_breakdown = self._compute_domain_breakdown(results)
        error_taxonomy = self.compute_formalization_error_taxonomy(results)

        return EvaluationReport(
            generation=generation_metrics,
            filtering=filtering_metrics,
            formalization=formalization_metrics,
            proof_search=proof_search_metrics,
            end_to_end=end_to_end_metrics,
            layer_breakdown=layer_breakdown,
            domain_breakdown=domain_breakdown,
            error_taxonomy=error_taxonomy,
            mean_formalization_retries=mean_retries,
            mean_proof_duration_seconds=mean_duration,
        )

    def print_report(self, report: EvaluationReport) -> None:
        """
        Print a formatted evaluation report to stdout.

        Uses :mod:`rich` tables when available; falls back to plain-text
        ASCII tables otherwise.

        Args:
            report: Populated :class:`EvaluationReport` to render.
        """
        try:
            self._print_rich(report)
        except ImportError:
            self._print_plain(report)

    def save_report(self, report: EvaluationReport, path: Path) -> None:
        """
        Persist the evaluation report as both a JSON file and a Markdown file.

        Creates the parent directory if it does not exist.

        Args:
            report: Populated :class:`EvaluationReport` to save.
            path: Base path (without extension); ``.json`` and ``.md``
                suffixes are appended automatically.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        json_path = path.with_suffix(".json")
        md_path = path.with_suffix(".md")

        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(report.to_markdown_table(), encoding="utf-8")

        logger.info("Evaluation report saved to %s and %s", json_path, md_path)

    # ------------------------------------------------------------------
    # Breakdown computations
    # ------------------------------------------------------------------

    def compute_formalization_error_taxonomy(
        self, results: list[PipelineResult]
    ) -> dict[str, int]:
        """
        Classify Lean 4 formalization errors into semantic categories.

        Error strings from ``error_history`` of unformalizable conjectures
        are matched against pattern heuristics.  Each error string is
        assigned to exactly one category (the first match wins).

        Categories:
            - ``unknown_identifier`` — references to undefined names
            - ``type_mismatch`` — type-checking failures
            - ``ambiguous`` — ambiguous overloaded terms
            - ``missing_instance`` — failed typeclass instance search
            - ``universe`` — universe polymorphism errors
            - ``syntax`` — parse errors
            - ``noncomputable`` — noncomputable definition errors
            - ``other`` — everything else

        Args:
            results: Full list of pipeline results.

        Returns:
            Dict mapping category name to occurrence count.
        """
        taxonomy: dict[str, int] = defaultdict(int)

        _patterns: list[tuple[str, re.Pattern]] = [
            ("unknown_identifier", re.compile(r"unknown identifier|unknown constant|undeclared", re.I)),
            ("type_mismatch", re.compile(r"type mismatch|expected type|has type", re.I)),
            ("ambiguous", re.compile(r"ambiguous", re.I)),
            ("missing_instance", re.compile(r"failed to synthesize|no instance found|instance", re.I)),
            ("universe", re.compile(r"universe|Sort|Type \d", re.I)),
            ("syntax", re.compile(r"expected token|unexpected token|parse error|':='", re.I)),
            ("noncomputable", re.compile(r"noncomputable", re.I)),
        ]

        for result in results:
            if result.formalization is None:
                continue
            if result.formalization.status is not FormalizationStatus.UNFORMALIZABLE:
                continue

            for error_msg in result.formalization.error_history:
                matched = False
                for category, pattern in _patterns:
                    if pattern.search(error_msg):
                        taxonomy[category] += 1
                        matched = True
                        break
                if not matched:
                    taxonomy["other"] += 1

        return dict(taxonomy)

    def compute_layer_breakdown(self, results: list[PipelineResult]) -> dict[str, int]:
        """
        Count the number of proofs closed by each proof-search layer.

        Args:
            results: Full list of pipeline results.

        Returns:
            Dict mapping :class:`~conjlean.schemas.ProofLayer` value strings
            to proof counts.
        """
        breakdown: dict[str, int] = {layer.value: 0 for layer in ProofLayer}

        for result in results:
            if (
                result.proof is not None
                and result.proof.status is ProofStatus.PROVED
                and result.proof.layer is not None
            ):
                breakdown[result.proof.layer.value] += 1

        return breakdown

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_domain_breakdown(
        self, results: list[PipelineResult]
    ) -> dict[str, dict[str, int]]:
        """
        Build a nested map of ``domain → {pipeline_status: count}``.

        Args:
            results: Full list of pipeline results.

        Returns:
            Nested dict keyed by domain value and pipeline status value.
        """
        breakdown: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for result in results:
            domain = result.conjecture.domain.value
            status = result.final_status.value
            breakdown[domain][status] += 1

        return {d: dict(counts) for d, counts in breakdown.items()}

    def _print_rich(self, report: EvaluationReport) -> None:
        """
        Render the report using :mod:`rich` Console and Table.

        Args:
            report: Report to render.

        Raises:
            ImportError: If :mod:`rich` is not installed (caller catches this).
        """
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()

        # Pipeline stage table
        stage_table = Table(
            title="Pipeline Stage Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        stage_table.add_column("Stage", style="bold")
        stage_table.add_column("Total", justify="right")
        stage_table.add_column("Success", justify="right")
        stage_table.add_column("Rate", justify="right")
        stage_table.add_column("95% CI", justify="right")

        stage_rows = [
            ("Generation", report.generation),
            ("Filtering", report.filtering),
            ("Formalization", report.formalization),
            ("Proof Search", report.proof_search),
            ("End-to-End", report.end_to_end),
        ]
        for name, m in stage_rows:
            ci_str = (
                f"[{m.ci_lower:.1%}, {m.ci_upper:.1%}]"
                if m.ci_lower is not None
                else "—"
            )
            stage_table.add_row(
                name,
                str(m.total),
                str(m.success),
                f"{m.rate:.1%}",
                ci_str,
            )
        console.print(stage_table)

        # Layer breakdown
        if any(v > 0 for v in report.layer_breakdown.values()):
            layer_table = Table(
                title="Proof Layer Breakdown",
                box=box.SIMPLE,
                header_style="bold magenta",
            )
            layer_table.add_column("Layer")
            layer_table.add_column("Count", justify="right")
            for layer, count in sorted(report.layer_breakdown.items()):
                if count > 0:
                    layer_table.add_row(layer, str(count))
            console.print(layer_table)

        # Error taxonomy
        if report.error_taxonomy:
            err_table = Table(
                title="Formalization Error Taxonomy",
                box=box.SIMPLE,
                header_style="bold red",
            )
            err_table.add_column("Error Type")
            err_table.add_column("Count", justify="right")
            for err_type, count in sorted(
                report.error_taxonomy.items(), key=lambda x: -x[1]
            ):
                err_table.add_row(err_type, str(count))
            console.print(err_table)

        console.print(
            f"\n[bold]Mean formalization retries:[/bold] "
            f"{report.mean_formalization_retries:.2f}  |  "
            f"[bold]Mean proof duration:[/bold] "
            f"{report.mean_proof_duration_seconds:.2f}s"
        )

    def _print_plain(self, report: EvaluationReport) -> None:
        """
        Render the report as plain-text ASCII tables (no dependencies).

        Args:
            report: Report to render.
        """
        def _row(cols: list[str], widths: list[int]) -> str:
            return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"

        def _sep(widths: list[int]) -> str:
            return "+-" + "-+-".join("-" * w for w in widths) + "-+"

        print("\n=== ConjLean Evaluation Report ===\n")

        headers = ["Stage", "Total", "Success", "Rate", "95% CI"]
        _stage_ms = [
            ("Generation", report.generation),
            ("Filtering", report.filtering),
            ("Formalization", report.formalization),
            ("Proof Search", report.proof_search),
            ("End-to-End", report.end_to_end),
        ]
        rows = [
            [
                name,
                str(m.total),
                str(m.success),
                f"{m.rate:.1%}",
                (f"[{m.ci_lower:.1%}, {m.ci_upper:.1%}]" if m.ci_lower is not None else "—"),
            ]
            for name, m in _stage_ms
        ]

        widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        print(_sep(widths))
        print(_row(headers, widths))
        print(_sep(widths))
        for row in rows:
            print(_row(row, widths))
        print(_sep(widths))

        if report.layer_breakdown:
            print("\nProof Layer Breakdown:")
            for layer, count in sorted(report.layer_breakdown.items()):
                if count > 0:
                    print(f"  {layer}: {count}")

        if report.error_taxonomy:
            print("\nFormalization Error Taxonomy:")
            for err_type, count in sorted(
                report.error_taxonomy.items(), key=lambda x: -x[1]
            ):
                print(f"  {err_type}: {count}")

        print(
            f"\nMean formalization retries: {report.mean_formalization_retries:.2f}"
            f"  |  Mean proof duration: {report.mean_proof_duration_seconds:.2f}s\n"
        )
