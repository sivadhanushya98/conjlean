"""
Tests for conjlean.multi_seed.

All tests are pure-computation and require no LLM, Lean, or SymPy runtime.
"""

from __future__ import annotations

import csv
import math

import pytest

from conjlean.multi_seed import (
    AggregatedMetrics,
    MultiSeedAggregator,
    SeedConfig,
    SeedRunMetrics,
    _compute_ci,
    _t_critical,
    metrics_from_evaluation_report,
    metrics_from_refute_metrics,
)


# ---------------------------------------------------------------------------
# SeedConfig
# ---------------------------------------------------------------------------


class TestSeedConfig:
    def test_seeds_length(self) -> None:
        cfg = SeedConfig(n_seeds=5)
        assert len(cfg.seeds) == 5

    def test_seeds_deterministic(self) -> None:
        assert SeedConfig(n_seeds=10, base_seed=0).seeds == SeedConfig(n_seeds=10, base_seed=0).seeds

    def test_different_base_seeds_differ(self) -> None:
        s0 = SeedConfig(n_seeds=5, base_seed=0).seeds
        s1 = SeedConfig(n_seeds=5, base_seed=1).seeds
        assert s0 != s1

    def test_seeds_are_non_negative(self) -> None:
        for s in SeedConfig(n_seeds=20, base_seed=42).seeds:
            assert s >= 0

    def test_seeds_are_integers(self) -> None:
        for s in SeedConfig(n_seeds=5).seeds:
            assert isinstance(s, int)

    def test_n_seeds_one(self) -> None:
        cfg = SeedConfig(n_seeds=1, base_seed=7)
        assert len(cfg.seeds) == 1

    def test_n_seeds_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            SeedConfig(n_seeds=0)

    def test_negative_base_seed_raises(self) -> None:
        with pytest.raises(ValueError):
            SeedConfig(n_seeds=3, base_seed=-1)

    def test_seeds_unique(self) -> None:
        seeds = SeedConfig(n_seeds=100, base_seed=0).seeds
        assert len(set(seeds)) == len(seeds)

    def test_larger_n_extends_smaller(self) -> None:
        s5 = SeedConfig(n_seeds=5, base_seed=0).seeds
        s10 = SeedConfig(n_seeds=10, base_seed=0).seeds
        assert s10[:5] == s5


# ---------------------------------------------------------------------------
# _t_critical
# ---------------------------------------------------------------------------


class TestTCritical:
    def test_df1_large(self) -> None:
        t = _t_critical(df=1)
        assert t == pytest.approx(12.706, abs=1e-3)

    def test_df10(self) -> None:
        t = _t_critical(df=10)
        assert t == pytest.approx(2.228, abs=1e-3)

    def test_large_df_approaches_z(self) -> None:
        t = _t_critical(df=100)
        assert t == pytest.approx(1.960, abs=0.05)

    def test_decreasing_with_df(self) -> None:
        vals = [_t_critical(df=d) for d in [2, 5, 10, 30]]
        assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# _compute_ci
# ---------------------------------------------------------------------------


class TestComputeCI:
    def test_returns_none_for_single_value(self) -> None:
        assert _compute_ci([0.5]) is None

    def test_symmetric_around_mean(self) -> None:
        vals = [0.6, 0.7, 0.8]
        lo, hi = _compute_ci(vals)
        mean = sum(vals) / len(vals)
        assert lo == pytest.approx(mean - (mean - lo), abs=1e-10)
        assert hi == pytest.approx(mean + (hi - mean), abs=1e-10)

    def test_zero_variance_gives_zero_width(self) -> None:
        lo, hi = _compute_ci([0.75, 0.75, 0.75])
        assert lo == pytest.approx(0.75)
        assert hi == pytest.approx(0.75)

    def test_wider_for_higher_variance(self) -> None:
        narrow_lo, narrow_hi = _compute_ci([0.5, 0.5, 0.5, 0.6, 0.4])
        wide_lo, wide_hi = _compute_ci([0.1, 0.5, 0.9, 0.5, 0.5])
        assert (wide_hi - wide_lo) > (narrow_hi - narrow_lo)

    def test_bounds_ordered(self) -> None:
        lo, hi = _compute_ci([0.2, 0.8, 0.5, 0.3, 0.7])
        assert lo <= hi

    def test_mean_within_ci(self) -> None:
        vals = [0.3, 0.5, 0.7]
        lo, hi = _compute_ci(vals)
        mean = sum(vals) / 3
        assert lo <= mean <= hi


# ---------------------------------------------------------------------------
# SeedRunMetrics
# ---------------------------------------------------------------------------


class TestSeedRunMetrics:
    def test_construction(self) -> None:
        m = SeedRunMetrics(seed=42, run_index=0, metrics={"rate": 0.8})
        assert m.seed == 42
        assert m.run_index == 0
        assert m.metrics["rate"] == 0.8

    def test_default_empty_metrics(self) -> None:
        m = SeedRunMetrics(seed=1, run_index=0)
        assert m.metrics == {}


# ---------------------------------------------------------------------------
# MultiSeedAggregator.aggregate
# ---------------------------------------------------------------------------


def _make_runs(values_list: list[dict[str, float]]) -> list[SeedRunMetrics]:
    seeds = SeedConfig(n_seeds=len(values_list), base_seed=99).seeds
    return [
        SeedRunMetrics(seed=seeds[i], run_index=i, metrics=v)
        for i, v in enumerate(values_list)
    ]


class TestMultiSeedAggregatorAggregate:
    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            MultiSeedAggregator().aggregate([])

    def test_raises_on_inconsistent_keys(self) -> None:
        runs = [
            SeedRunMetrics(seed=1, run_index=0, metrics={"a": 0.5}),
            SeedRunMetrics(seed=2, run_index=1, metrics={"b": 0.5}),
        ]
        with pytest.raises(ValueError):
            MultiSeedAggregator().aggregate(runs)

    def test_single_run_zero_std(self) -> None:
        runs = [SeedRunMetrics(seed=42, run_index=0, metrics={"rate": 0.8})]
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.mean["rate"] == pytest.approx(0.8)
        assert agg.std["rate"] == pytest.approx(0.0)

    def test_mean_correct(self) -> None:
        runs = _make_runs([{"rate": 0.6}, {"rate": 0.7}, {"rate": 0.8}])
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.mean["rate"] == pytest.approx(0.7)

    def test_std_correct(self) -> None:
        runs = _make_runs([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.std["x"] == pytest.approx(1.0)

    def test_uniform_runs_zero_std(self) -> None:
        runs = _make_runs([{"f1": 0.9}] * 5)
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.std["f1"] == pytest.approx(0.0)
        assert agg.ci_lower["f1"] == pytest.approx(0.9)
        assert agg.ci_upper["f1"] == pytest.approx(0.9)

    def test_n_seeds_correct(self) -> None:
        runs = _make_runs([{"r": 0.5}] * 7)
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.n_seeds == 7

    def test_metric_names_sorted(self) -> None:
        runs = _make_runs([{"z": 0.1, "a": 0.2, "m": 0.3}])
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.metric_names == sorted(agg.metric_names)

    def test_ci_bounds_ordered(self) -> None:
        runs = _make_runs([{"r": v} for v in [0.3, 0.5, 0.7, 0.6, 0.4]])
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.ci_lower["r"] <= agg.mean["r"] <= agg.ci_upper["r"]

    def test_seeds_preserved(self) -> None:
        runs = _make_runs([{"x": i * 0.1} for i in range(4)])
        agg = MultiSeedAggregator().aggregate(runs)
        assert agg.seeds == [r.seed for r in runs]

    def test_runs_preserved(self) -> None:
        runs = _make_runs([{"v": 0.5}] * 3)
        agg = MultiSeedAggregator().aggregate(runs)
        assert len(agg.runs) == 3

    def test_multiple_metrics(self) -> None:
        data = [{"precision": 0.8 + i * 0.05, "recall": 0.7 + i * 0.03} for i in range(4)]
        runs = _make_runs(data)
        agg = MultiSeedAggregator().aggregate(runs)
        assert "precision" in agg.mean
        assert "recall" in agg.mean
        expected_prec = sum(d["precision"] for d in data) / 4
        assert agg.mean["precision"] == pytest.approx(expected_prec)

    def test_custom_ci_level(self) -> None:
        runs = _make_runs([{"r": v} for v in [0.5, 0.6, 0.7]])
        agg_95 = MultiSeedAggregator().aggregate(runs, ci_level=0.95)
        agg_90 = MultiSeedAggregator().aggregate(runs, ci_level=0.90)
        width_95 = agg_95.ci_upper["r"] - agg_95.ci_lower["r"]
        width_90 = agg_90.ci_upper["r"] - agg_90.ci_lower["r"]
        assert width_95 >= width_90 - 1e-9


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestCSVExport:
    def _make_agg(self) -> AggregatedMetrics:
        runs = _make_runs([
            {"rate": 0.6, "f1": 0.7},
            {"rate": 0.7, "f1": 0.8},
            {"rate": 0.8, "f1": 0.9},
        ])
        return MultiSeedAggregator().aggregate(runs)

    def test_per_seed_csv_created(self, tmp_path) -> None:
        agg = self._make_agg()
        base = tmp_path / "out"
        MultiSeedAggregator().to_csv(agg, base)
        assert (tmp_path / "out_per_seed.csv").exists()

    def test_aggregate_csv_created(self, tmp_path) -> None:
        agg = self._make_agg()
        base = tmp_path / "out"
        MultiSeedAggregator().to_csv(agg, base)
        assert (tmp_path / "out_aggregate.csv").exists()

    def test_per_seed_row_count(self, tmp_path) -> None:
        agg = self._make_agg()
        base = tmp_path / "out"
        MultiSeedAggregator().to_csv(agg, base)
        with (tmp_path / "out_per_seed.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_per_seed_headers(self, tmp_path) -> None:
        agg = self._make_agg()
        base = tmp_path / "out"
        MultiSeedAggregator().to_csv(agg, base)
        with (tmp_path / "out_per_seed.csv").open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
        assert "seed" in headers
        assert "run_index" in headers
        assert "rate" in headers
        assert "f1" in headers

    def test_aggregate_csv_has_mean_row(self, tmp_path) -> None:
        agg = self._make_agg()
        base = tmp_path / "out"
        MultiSeedAggregator().to_csv(agg, base)
        with (tmp_path / "out_aggregate.csv").open() as f:
            rows = {row["stat"]: row for row in csv.DictReader(f)}
        assert "mean" in rows
        assert float(rows["mean"]["rate"]) == pytest.approx(0.7)

    def test_aggregate_csv_has_std_row(self, tmp_path) -> None:
        agg = self._make_agg()
        base = tmp_path / "out"
        MultiSeedAggregator().to_csv(agg, base)
        with (tmp_path / "out_aggregate.csv").open() as f:
            rows = {row["stat"]: row for row in csv.DictReader(f)}
        assert "std" in rows


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    def _make_agg(self) -> AggregatedMetrics:
        runs = _make_runs([{"precision": 0.9 - i * 0.05} for i in range(5)])
        return MultiSeedAggregator().aggregate(runs)

    def test_contains_metric_name(self) -> None:
        md = MultiSeedAggregator().to_markdown(self._make_agg())
        assert "precision" in md

    def test_contains_n_seeds(self) -> None:
        md = MultiSeedAggregator().to_markdown(self._make_agg())
        assert "N=5" in md

    def test_contains_ci_header(self) -> None:
        md = MultiSeedAggregator().to_markdown(self._make_agg())
        assert "95% CI" in md

    def test_contains_per_seed_section(self) -> None:
        md = MultiSeedAggregator().to_markdown(self._make_agg())
        assert "Per-Seed" in md

    def test_contains_seed_list(self) -> None:
        agg = self._make_agg()
        md = MultiSeedAggregator().to_markdown(agg)
        assert "Seed List" in md
        for s in agg.seeds:
            assert str(s) in md

    def test_save_all_creates_markdown(self, tmp_path) -> None:
        agg = self._make_agg()
        MultiSeedAggregator().save_all(agg, tmp_path / "summary")
        assert (tmp_path / "summary_summary.md").exists()

    def test_save_all_creates_both_csvs(self, tmp_path) -> None:
        agg = self._make_agg()
        MultiSeedAggregator().save_all(agg, tmp_path / "summary")
        assert (tmp_path / "summary_per_seed.csv").exists()
        assert (tmp_path / "summary_aggregate.csv").exists()


# ---------------------------------------------------------------------------
# metrics_from_evaluation_report
# ---------------------------------------------------------------------------


class TestMetricsFromEvaluationReport:
    def _make_report(self):
        from conjlean.evaluate import Evaluator, StageMetrics, EvaluationReport

        def sm(total, success):
            return StageMetrics(
                total=total,
                success=success,
                rate=success / max(total, 1),
            )

        return EvaluationReport(
            generation=sm(100, 100),
            filtering=sm(100, 80),
            formalization=sm(80, 60),
            proof_search=sm(60, 40),
            end_to_end=sm(100, 40),
            mean_formalization_retries=2.5,
            mean_proof_duration_seconds=1.3,
        )

    def test_returns_seed_run_metrics(self) -> None:
        report = self._make_report()
        m = metrics_from_evaluation_report(report, seed=42, run_index=0)
        assert isinstance(m, SeedRunMetrics)
        assert m.seed == 42
        assert m.run_index == 0

    def test_filtering_rate_correct(self) -> None:
        report = self._make_report()
        m = metrics_from_evaluation_report(report, seed=0, run_index=0)
        assert m.metrics["filtering_rate"] == pytest.approx(0.8)

    def test_end_to_end_rate_correct(self) -> None:
        report = self._make_report()
        m = metrics_from_evaluation_report(report, seed=0, run_index=0)
        assert m.metrics["end_to_end_rate"] == pytest.approx(0.4)

    def test_mean_retries_included(self) -> None:
        report = self._make_report()
        m = metrics_from_evaluation_report(report, seed=0, run_index=0)
        assert m.metrics["mean_retries"] == pytest.approx(2.5)

    def test_all_expected_keys_present(self) -> None:
        report = self._make_report()
        m = metrics_from_evaluation_report(report, seed=0, run_index=0)
        for key in (
            "filtering_rate",
            "formalization_rate",
            "proof_search_rate",
            "end_to_end_rate",
            "mean_retries",
            "mean_proof_duration_s",
        ):
            assert key in m.metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# metrics_from_refute_metrics
# ---------------------------------------------------------------------------


class TestMetricsFromRefuteMetrics:
    def _make_refute_metrics(self):
        from conjlean.refute_evaluate import RefuteMetrics

        return RefuteMetrics(
            n_total=50,
            n_refuted=35,
            n_survived=15,
            n_refined=5,
            precision=0.85,
            recall=0.78,
            false_positive_rate=0.12,
            f1=0.81,
            mean_rounds=3.2,
            strategy_breakdown={},
            domain_breakdown={},
            tier_breakdown={},
            refinement_rate=0.14,
            mean_wall_seconds_per_conjecture=4.5,
        )

    def test_returns_seed_run_metrics(self) -> None:
        m = metrics_from_refute_metrics(self._make_refute_metrics(), seed=7, run_index=2)
        assert isinstance(m, SeedRunMetrics)
        assert m.seed == 7
        assert m.run_index == 2

    def test_precision_correct(self) -> None:
        m = metrics_from_refute_metrics(self._make_refute_metrics(), seed=0, run_index=0)
        assert m.metrics["precision"] == pytest.approx(0.85)

    def test_f1_correct(self) -> None:
        m = metrics_from_refute_metrics(self._make_refute_metrics(), seed=0, run_index=0)
        assert m.metrics["f1"] == pytest.approx(0.81)

    def test_all_expected_keys_present(self) -> None:
        m = metrics_from_refute_metrics(self._make_refute_metrics(), seed=0, run_index=0)
        for key in (
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "n_total",
            "n_refuted",
            "refinement_rate",
            "mean_rounds",
            "mean_wall_seconds",
        ):
            assert key in m.metrics, f"Missing key: {key}"

    def test_n_total_float(self) -> None:
        m = metrics_from_refute_metrics(self._make_refute_metrics(), seed=0, run_index=0)
        assert isinstance(m.metrics["n_total"], float)


# ---------------------------------------------------------------------------
# End-to-end aggregation
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pipeline_aggregate_roundtrip(self) -> None:
        from conjlean.evaluate import EvaluationReport, StageMetrics

        def sm(t, s):
            return StageMetrics(total=t, success=s, rate=s / max(t, 1))

        reports = [
            EvaluationReport(
                generation=sm(100, 100),
                filtering=sm(100, 80 + i * 2),
                formalization=sm(80 + i * 2, 60 + i),
                proof_search=sm(60 + i, 40),
                end_to_end=sm(100, 40),
                mean_formalization_retries=float(i),
                mean_proof_duration_seconds=1.0,
            )
            for i in range(5)
        ]

        cfg = SeedConfig(n_seeds=5, base_seed=0)
        runs = [
            metrics_from_evaluation_report(r, seed=cfg.seeds[i], run_index=i)
            for i, r in enumerate(reports)
        ]
        agg = MultiSeedAggregator().aggregate(runs)

        assert agg.n_seeds == 5
        assert "filtering_rate" in agg.mean
        assert 0.0 <= agg.ci_lower["filtering_rate"] <= agg.ci_upper["filtering_rate"] <= 1.0

    def test_refute_aggregate_roundtrip(self) -> None:
        from conjlean.refute_evaluate import RefuteMetrics

        metrics_list = [
            RefuteMetrics(
                n_total=60,
                n_refuted=40 + i,
                n_survived=20 - i,
                n_refined=i,
                precision=0.80 + i * 0.02,
                recall=0.75 + i * 0.01,
                false_positive_rate=0.10 - i * 0.01,
                f1=0.77 + i * 0.015,
                mean_rounds=3.0,
                strategy_breakdown={},
                domain_breakdown={},
                tier_breakdown={},
                refinement_rate=0.05,
            )
            for i in range(5)
        ]

        cfg = SeedConfig(n_seeds=5, base_seed=7)
        runs = [
            metrics_from_refute_metrics(m, seed=cfg.seeds[i], run_index=i)
            for i, m in enumerate(metrics_list)
        ]
        agg = MultiSeedAggregator().aggregate(runs)

        assert math.isfinite(agg.mean["f1"])
        assert agg.ci_lower["f1"] <= agg.mean["f1"] <= agg.ci_upper["f1"]
