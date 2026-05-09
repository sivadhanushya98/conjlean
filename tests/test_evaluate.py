"""
Tests for conjlean.evaluate — bootstrap_ci and Evaluator.

Uses minimal PipelineResult stubs so no LLM or Lean runtime is needed.
"""

from __future__ import annotations

import pytest

from conjlean.evaluate import Evaluator, StageMetrics, bootstrap_ci
from conjlean.schemas import (
    Conjecture,
    Domain,
    FilterResult,
    FilterStatus,
    FormalizedConjecture,
    FormalizationStatus,
    PipelineResult,
    PipelineStatus,
    ProofResult,
    ProofStatus,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_none_below_threshold(self) -> None:
        assert bootstrap_ci([True, False, True, False]) is None

    def test_returns_tuple_above_threshold(self) -> None:
        outcomes = [True] * 10 + [False] * 10
        result = bootstrap_ci(outcomes)
        assert result is not None
        lo, hi = result
        assert 0.0 <= lo <= hi <= 1.0

    def test_all_true_ci_near_one(self) -> None:
        result = bootstrap_ci([True] * 50)
        assert result is not None
        lo, hi = result
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_all_false_ci_near_zero(self) -> None:
        result = bootstrap_ci([False] * 50)
        assert result is not None
        lo, hi = result
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_ci_bounds_ordered(self) -> None:
        outcomes = [True] * 30 + [False] * 70
        lo, hi = bootstrap_ci(outcomes)
        assert lo <= hi

    def test_reproducible_with_same_seed(self) -> None:
        outcomes = [True, False] * 25
        r1 = bootstrap_ci(outcomes, seed=7)
        r2 = bootstrap_ci(outcomes, seed=7)
        assert r1 == r2

    def test_exactly_min_bootstrap_n(self) -> None:
        # _MIN_BOOTSTRAP_N is 5; exactly 5 should return a result
        result = bootstrap_ci([True, False, True, False, True])
        assert result is not None

    def test_ci_width_zero_for_constant(self) -> None:
        lo, hi = bootstrap_ci([True] * 20)
        assert hi - lo == pytest.approx(0.0)

    def test_90_pct_level_narrower_than_95(self) -> None:
        outcomes = [True] * 40 + [False] * 60
        lo_95, hi_95 = bootstrap_ci(outcomes, ci_level=0.95)
        lo_90, hi_90 = bootstrap_ci(outcomes, ci_level=0.90)
        assert (hi_90 - lo_90) <= (hi_95 - lo_95) + 1e-9


# ---------------------------------------------------------------------------
# PipelineResult factory helpers
# ---------------------------------------------------------------------------


def _conjecture(cid: str = "c1", domain: Domain = Domain.NUMBER_THEORY) -> Conjecture:
    return Conjecture(id=cid, domain=domain, nl_statement="test", variables=["n"])


def _surviving_filter(conj: Conjecture) -> FilterResult:
    return FilterResult(conjecture=conj, status=FilterStatus.SURVIVING)


def _disproved_filter(conj: Conjecture) -> FilterResult:
    return FilterResult(
        conjecture=conj,
        status=FilterStatus.DISPROVED,
        counterexample="n=1",
    )


def _typechecks_form(conj: Conjecture) -> FormalizedConjecture:
    return FormalizedConjecture(
        conjecture=conj,
        lean_code="theorem t : True := trivial",
        status=FormalizationStatus.TYPECHECKS,
        retries=0,
    )


def _failed_form(conj: Conjecture) -> FormalizedConjecture:
    return FormalizedConjecture(
        conjecture=conj,
        lean_code="",
        status=FormalizationStatus.UNFORMALIZABLE,
        retries=1,
        error_history=["unknown identifier"],
    )


def _proved_proof(form: FormalizedConjecture) -> ProofResult:
    return ProofResult(
        formalized=form,
        status=ProofStatus.PROVED,
        proof="exact trivial",
        duration_seconds=0.5,
    )


def _open_proof(form: FormalizedConjecture) -> ProofResult:
    return ProofResult(
        formalized=form,
        status=ProofStatus.OPEN,
        duration_seconds=1.0,
    )


def _make_result(
    cid: str = "c1",
    domain: Domain = Domain.NUMBER_THEORY,
    filter_status: FilterStatus = FilterStatus.SURVIVING,
    form_status: FormalizationStatus | None = FormalizationStatus.TYPECHECKS,
    proof_status: ProofStatus | None = ProofStatus.PROVED,
) -> PipelineResult:
    conj = _conjecture(cid=cid, domain=domain)

    fr = (
        FilterResult(conjecture=conj, status=filter_status)
        if filter_status is not None
        else None
    )

    form: FormalizedConjecture | None = None
    if form_status is not None and filter_status is FilterStatus.SURVIVING:
        form = (
            _typechecks_form(conj)
            if form_status is FormalizationStatus.TYPECHECKS
            else _failed_form(conj)
        )

    proof: ProofResult | None = None
    if proof_status is not None and form is not None and form.status is FormalizationStatus.TYPECHECKS:
        proof = (
            _proved_proof(form)
            if proof_status is ProofStatus.PROVED
            else _open_proof(form)
        )

    if proof is not None and proof.status is ProofStatus.PROVED:
        final = PipelineStatus.PROVED
    elif form is not None and form.status is FormalizationStatus.TYPECHECKS:
        final = PipelineStatus.OPEN
    elif filter_status is FilterStatus.DISPROVED:
        final = PipelineStatus.DISPROVED
    else:
        final = PipelineStatus.FAILED

    return PipelineResult(
        conjecture=conj,
        filter_result=fr,
        formalization=form,
        proof=proof,
        final_status=final,
    )


# ---------------------------------------------------------------------------
# Evaluator.evaluate — basic sanity
# ---------------------------------------------------------------------------


class TestEvaluatorBasic:
    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            Evaluator().evaluate([])

    def test_single_proved(self) -> None:
        r = _make_result(proof_status=ProofStatus.PROVED)
        report = Evaluator().evaluate([r])
        assert report.end_to_end.success == 1
        assert report.end_to_end.rate == pytest.approx(1.0)

    def test_single_disproved(self) -> None:
        r = _make_result(
            filter_status=FilterStatus.DISPROVED,
            form_status=None,
            proof_status=None,
        )
        report = Evaluator().evaluate([r])
        assert report.filtering.success == 0
        assert report.end_to_end.success == 0

    def test_rates_sum_correctly(self) -> None:
        proved = _make_result("c1", proof_status=ProofStatus.PROVED)
        disproved = _make_result(
            "c2",
            filter_status=FilterStatus.DISPROVED,
            form_status=None,
            proof_status=None,
        )
        report = Evaluator().evaluate([proved, disproved])
        assert report.generation.total == 2
        assert report.filtering.success == 1
        assert report.end_to_end.success == 1
        assert report.end_to_end.rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CI fields on StageMetrics
# ---------------------------------------------------------------------------


class TestStageMetricsCI:
    def test_ci_defaults_to_none(self) -> None:
        m = StageMetrics(total=10, success=5, rate=0.5)
        assert m.ci_lower is None
        assert m.ci_upper is None

    def test_ci_populated_for_large_batch(self) -> None:
        results = [_make_result(f"c{i}", proof_status=ProofStatus.PROVED) for i in range(20)]
        report = Evaluator().evaluate(results)
        assert report.filtering.ci_lower is not None
        assert report.filtering.ci_upper is not None

    def test_ci_bounds_valid(self) -> None:
        results = [
            _make_result(
                f"c{i}",
                proof_status=(ProofStatus.PROVED if i % 2 == 0 else ProofStatus.OPEN),
            )
            for i in range(20)
        ]
        report = Evaluator().evaluate(results)
        for m in (report.filtering, report.end_to_end):
            if m.ci_lower is not None:
                assert 0.0 <= m.ci_lower <= m.ci_upper <= 1.0

    def test_to_dict_includes_ci_when_present(self) -> None:
        results = [_make_result(f"c{i}", proof_status=ProofStatus.PROVED) for i in range(20)]
        report = Evaluator().evaluate(results)
        d = report.to_dict()
        if report.filtering.ci_lower is not None:
            assert "ci_lower" in d["filtering"]
            assert "ci_upper" in d["filtering"]

    def test_to_dict_omits_ci_when_absent(self) -> None:
        m = StageMetrics(total=2, success=1, rate=0.5)
        assert m.ci_lower is None
        d: dict = {
            "total": m.total,
            "success": m.success,
            "rate": m.rate,
            "breakdown": m.breakdown,
        }
        if m.ci_lower is not None:
            d["ci_lower"] = m.ci_lower
        assert "ci_lower" not in d

    def test_markdown_table_has_ci_column(self) -> None:
        results = [_make_result(f"c{i}", proof_status=ProofStatus.PROVED) for i in range(20)]
        report = Evaluator().evaluate(results)
        md = report.to_markdown_table()
        assert "95% CI" in md

    def test_generation_ci_valid_when_present(self) -> None:
        results = [_make_result(f"c{i}") for i in range(20)]
        report = Evaluator().evaluate(results)
        if report.generation.ci_lower is not None:
            assert report.generation.ci_lower == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Markdown and serialisation
# ---------------------------------------------------------------------------


class TestEvaluatorReports:
    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        results = [_make_result(f"c{i}") for i in range(10)]
        report = Evaluator().evaluate(results)
        blob = json.dumps(report.to_dict())
        d = json.loads(blob)
        assert "filtering" in d
        assert "end_to_end" in d

    def test_markdown_table_contains_stage_names(self) -> None:
        results = [_make_result(f"c{i}") for i in range(5)]
        report = Evaluator().evaluate(results)
        md = report.to_markdown_table()
        for name in ("Generation", "Filtering", "Formalization", "Proof Search", "End-to-End"):
            assert name in md

    def test_print_plain_runs(self, capsys) -> None:
        results = [_make_result(f"c{i}") for i in range(3)]
        report = Evaluator().evaluate(results)
        Evaluator()._print_plain(report)
        captured = capsys.readouterr()
        assert "Stage" in captured.out
