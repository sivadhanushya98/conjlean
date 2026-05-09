"""
Tests for conjlean.refute_evaluate — _bootstrap_refute_ci and RefuteEvaluator.

Uses minimal stub objects; no LLM or SymPy runtime needed.
"""

from __future__ import annotations

import pytest

from conjlean.refute_evaluate import RefuteEvaluator, RefuteMetrics, _bootstrap_refute_ci
from conjlean.schemas import (
    BenchmarkEntry,
    BenchmarkTier,
    Conjecture,
    Domain,
    RefuteLoopResult,
    RefuteLoopStatus,
)


# ---------------------------------------------------------------------------
# _bootstrap_refute_ci
# ---------------------------------------------------------------------------


class TestBootstrapRefuteCI:
    def test_returns_none_below_threshold(self) -> None:
        pairs: list[tuple[bool, bool]] = [
            (True, True), (False, True), (True, False)
        ]
        result = _bootstrap_refute_ci(pairs)
        assert result["precision"] is None
        assert result["recall"] is None
        assert result["f1"] is None

    def test_returns_intervals_above_threshold(self) -> None:
        pairs = [(True, True)] * 5 + [(False, False)] * 5
        result = _bootstrap_refute_ci(pairs)
        for key in ("precision", "recall", "f1"):
            assert result[key] is not None
            lo, hi = result[key]
            assert 0.0 <= lo <= hi <= 1.0

    def test_all_true_positives(self) -> None:
        pairs = [(True, True)] * 20
        result = _bootstrap_refute_ci(pairs)
        assert result["precision"] is not None
        lo_p, hi_p = result["precision"]
        lo_r, hi_r = result["recall"]
        lo_f, hi_f = result["f1"]
        assert lo_p == pytest.approx(1.0)
        assert hi_p == pytest.approx(1.0)
        assert lo_r == pytest.approx(1.0)
        assert hi_r == pytest.approx(1.0)
        assert lo_f == pytest.approx(1.0)
        assert hi_f == pytest.approx(1.0)

    def test_no_truly_false_conjectures(self) -> None:
        # All refuted but none truly false → recall denominator = 0
        pairs = [(True, False)] * 20
        result = _bootstrap_refute_ci(pairs)
        assert result["recall"] is not None
        lo_r, hi_r = result["recall"]
        assert lo_r == pytest.approx(0.0)
        assert hi_r == pytest.approx(0.0)

    def test_reproducible_with_same_seed(self) -> None:
        pairs = [(True, True), (False, True), (True, False), (False, False)] * 5
        r1 = _bootstrap_refute_ci(pairs, seed=42)
        r2 = _bootstrap_refute_ci(pairs, seed=42)
        assert r1["precision"] == r2["precision"]
        assert r1["recall"] == r2["recall"]
        assert r1["f1"] == r2["f1"]

    def test_ci_bounds_ordered(self) -> None:
        pairs = [(i % 2 == 0, i % 3 == 0) for i in range(30)]
        result = _bootstrap_refute_ci(pairs)
        for key in ("precision", "recall", "f1"):
            if result[key] is not None:
                lo, hi = result[key]
                assert lo <= hi


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------


def _conj(cid: str, domain: Domain = Domain.NUMBER_THEORY) -> Conjecture:
    return Conjecture(id=cid, domain=domain, nl_statement="test", variables=["n"])


def _benchmark_entry(
    cid: str,
    ground_truth: str = "false",
    tier: BenchmarkTier = BenchmarkTier.TIER1_SYNTHETIC,
    domain: Domain = Domain.NUMBER_THEORY,
) -> BenchmarkEntry:
    return BenchmarkEntry(
        id=f"be_{cid}",
        tier=tier,
        conjecture=_conj(cid, domain),
        ground_truth_status=ground_truth,
        source="test",
    )


def _loop_result(
    cid: str,
    status: RefuteLoopStatus = RefuteLoopStatus.REFUTED,
    total_rounds: int = 1,
) -> RefuteLoopResult:
    return RefuteLoopResult(
        original_conjecture=_conj(cid),
        status=status,
        total_rounds=total_rounds,
    )


def _make_batch(
    n: int, refuted_mask: list[bool], truly_false_mask: list[bool]
) -> tuple[list[RefuteLoopResult], list[BenchmarkEntry]]:
    loop_results = [
        _loop_result(
            f"c{i}",
            status=(
                RefuteLoopStatus.REFUTED if refuted_mask[i] else RefuteLoopStatus.BUDGET_EXHAUSTED
            ),
        )
        for i in range(n)
    ]
    benchmark_entries = [
        _benchmark_entry(
            f"c{i}",
            ground_truth="false" if truly_false_mask[i] else "true",
        )
        for i in range(n)
    ]
    return loop_results, benchmark_entries


# ---------------------------------------------------------------------------
# RefuteEvaluator.evaluate — basic counts
# ---------------------------------------------------------------------------


class TestRefuteEvaluatorBasic:
    def test_raises_on_empty_loop_results(self) -> None:
        with pytest.raises(ValueError):
            RefuteEvaluator().evaluate([], [_benchmark_entry("c1")])

    def test_raises_on_empty_benchmark_entries(self) -> None:
        with pytest.raises(ValueError):
            RefuteEvaluator().evaluate([_loop_result("c1")], [])

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            RefuteEvaluator().evaluate(
                [_loop_result("c1"), _loop_result("c2")],
                [_benchmark_entry("c1")],
            )

    def test_perfect_precision_and_recall(self) -> None:
        n = 10
        lr, be = _make_batch(n, [True] * n, [True] * n)
        metrics = RefuteEvaluator().evaluate(lr, be)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(1.0)

    def test_zero_precision_when_all_false_positives(self) -> None:
        n = 10
        lr, be = _make_batch(n, [True] * n, [False] * n)
        metrics = RefuteEvaluator().evaluate(lr, be)
        assert metrics.precision == pytest.approx(0.0)
        assert metrics.recall == pytest.approx(0.0)

    def test_n_totals_correct(self) -> None:
        n = 8
        lr, be = _make_batch(n, [True] * 4 + [False] * 4, [True] * 8)
        metrics = RefuteEvaluator().evaluate(lr, be)
        assert metrics.n_total == n
        assert metrics.n_refuted == 4
        assert metrics.n_survived == 4

    def test_refinement_rate_zero_when_no_refinements(self) -> None:
        lr, be = _make_batch(5, [True] * 5, [True] * 5)
        metrics = RefuteEvaluator().evaluate(lr, be)
        assert metrics.refinement_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CI fields on RefuteMetrics
# ---------------------------------------------------------------------------


class TestRefuteMetricsCI:
    def test_ci_defaults_to_none(self) -> None:
        m = RefuteMetrics(
            n_total=0,
            n_refuted=0,
            n_survived=0,
            n_refined=0,
            precision=0.0,
            recall=0.0,
            false_positive_rate=0.0,
            f1=0.0,
            mean_rounds=0.0,
            strategy_breakdown={},
            domain_breakdown={},
            tier_breakdown={},
            refinement_rate=0.0,
        )
        assert m.precision_ci_lower is None
        assert m.recall_ci_lower is None
        assert m.f1_ci_lower is None

    def test_ci_populated_for_sufficient_sample(self) -> None:
        n = 20
        lr, be = _make_batch(n, [i % 2 == 0 for i in range(n)], [True] * n)
        metrics = RefuteEvaluator().evaluate(lr, be)
        assert metrics.precision_ci_lower is not None
        assert metrics.precision_ci_upper is not None
        assert metrics.recall_ci_lower is not None
        assert metrics.recall_ci_upper is not None
        assert metrics.f1_ci_lower is not None
        assert metrics.f1_ci_upper is not None

    def test_ci_bounds_valid(self) -> None:
        n = 20
        lr, be = _make_batch(n, [i % 2 == 0 for i in range(n)], [True] * n)
        metrics = RefuteEvaluator().evaluate(lr, be)
        for lo, hi in [
            (metrics.precision_ci_lower, metrics.precision_ci_upper),
            (metrics.recall_ci_lower, metrics.recall_ci_upper),
            (metrics.f1_ci_lower, metrics.f1_ci_upper),
        ]:
            if lo is not None:
                assert 0.0 <= lo <= hi <= 1.0

    def test_ci_none_for_tiny_batch(self) -> None:
        lr, be = _make_batch(3, [True, False, True], [True, True, False])
        metrics = RefuteEvaluator().evaluate(lr, be)
        assert metrics.precision_ci_lower is None

    def test_to_dict_includes_ci_when_present(self) -> None:
        n = 20
        lr, be = _make_batch(n, [i % 2 == 0 for i in range(n)], [True] * n)
        metrics = RefuteEvaluator().evaluate(lr, be)
        d = RefuteEvaluator._metrics_to_dict(metrics)
        if metrics.precision_ci_lower is not None:
            assert "precision_ci" in d
            assert "recall_ci" in d
            assert "f1_ci" in d
            assert len(d["precision_ci"]) == 2

    def test_to_dict_omits_ci_when_absent(self) -> None:
        m = RefuteMetrics(
            n_total=3,
            n_refuted=2,
            n_survived=1,
            n_refined=0,
            precision=0.5,
            recall=0.5,
            false_positive_rate=0.0,
            f1=0.5,
            mean_rounds=1.0,
            strategy_breakdown={},
            domain_breakdown={},
            tier_breakdown={},
            refinement_rate=0.0,
        )
        d = RefuteEvaluator._metrics_to_dict(m)
        assert "precision_ci" not in d

    def test_markdown_contains_ci_brackets(self) -> None:
        n = 20
        lr, be = _make_batch(n, [True] * n, [True] * n)
        metrics = RefuteEvaluator().evaluate(lr, be)
        md = RefuteEvaluator._metrics_to_markdown(metrics)
        if metrics.precision_ci_lower is not None:
            assert "[" in md and "]" in md

    def test_to_dict_json_serialisable(self) -> None:
        import json

        n = 20
        lr, be = _make_batch(
            n,
            [i % 3 != 0 for i in range(n)],
            [i % 2 == 0 for i in range(n)],
        )
        metrics = RefuteEvaluator().evaluate(lr, be)
        blob = json.dumps(RefuteEvaluator._metrics_to_dict(metrics))
        d = json.loads(blob)
        assert "precision" in d
        assert "n_total" in d


# ---------------------------------------------------------------------------
# print_report and save_report smoke tests
# ---------------------------------------------------------------------------


class TestRefuteEvaluatorReports:
    def test_print_report_runs(self) -> None:
        lr, be = _make_batch(5, [True] * 3 + [False] * 2, [True] * 5)
        metrics = RefuteEvaluator().evaluate(lr, be)
        RefuteEvaluator().print_report(metrics)  # should not raise

    def test_save_report_creates_files(self, tmp_path) -> None:
        lr, be = _make_batch(5, [True, False, True, False, True], [True] * 5)
        metrics = RefuteEvaluator().evaluate(lr, be)
        base = tmp_path / "test_report"
        RefuteEvaluator().save_report(metrics, base)
        assert (tmp_path / "test_report.json").exists()
        assert (tmp_path / "test_report.md").exists()
