"""
Tests for scripts/run_experiments.py — ExperimentRunner and ExperimentConfig.

All subprocess calls are mocked so no LLM, Lean, or actual pipeline execution
occurs.  _load_run_metrics is patched to return known SeedRunMetrics so tests
can verify aggregation logic in isolation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Import the module under test via importlib (it lives in scripts/, not a package)
import importlib.util, sys


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("run_experiments", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = mod  # must register before exec so @dataclass can resolve types
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_experiments.py"
    return _load_module(script)


# ---------------------------------------------------------------------------
# ExperimentConfig validation
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_valid_pipeline(self, mod, tmp_path) -> None:
        cfg = mod.ExperimentConfig(
            experiment_type="pipeline",
            config="configs/config.yaml",
            output_dir=tmp_path,
            n_seeds=3,
        )
        assert cfg.experiment_type == "pipeline"
        assert cfg.n_seeds == 3

    def test_valid_refute(self, mod, tmp_path) -> None:
        cfg = mod.ExperimentConfig(
            experiment_type="refute",
            config="configs/config.yaml",
            output_dir=tmp_path,
            n_seeds=2,
            benchmark_dir="data/benchmark",
        )
        assert cfg.experiment_type == "refute"

    def test_invalid_type_raises(self, mod, tmp_path) -> None:
        with pytest.raises(ValueError, match="experiment_type"):
            mod.ExperimentConfig(
                experiment_type="invalid",
                config="configs/config.yaml",
                output_dir=tmp_path,
                n_seeds=3,
            )

    def test_zero_seeds_raises(self, mod, tmp_path) -> None:
        with pytest.raises(ValueError):
            mod.ExperimentConfig(
                experiment_type="pipeline",
                config="configs/config.yaml",
                output_dir=tmp_path,
                n_seeds=0,
            )

    def test_refute_without_benchmark_raises(self, mod, tmp_path) -> None:
        with pytest.raises(ValueError, match="benchmark_dir"):
            mod.ExperimentConfig(
                experiment_type="refute",
                config="configs/config.yaml",
                output_dir=tmp_path,
                n_seeds=2,
                benchmark_dir=None,
            )

    def test_default_n_per_domain(self, mod, tmp_path) -> None:
        cfg = mod.ExperimentConfig(
            experiment_type="pipeline",
            config="c.yaml",
            output_dir=tmp_path,
            n_seeds=1,
        )
        assert cfg.n_per_domain == 100


# ---------------------------------------------------------------------------
# ExperimentRunner — directory and file helpers
# ---------------------------------------------------------------------------


def _make_runner(mod, tmp_path, *, n_seeds=3, exp_type="pipeline", **kwargs):
    cfg = mod.ExperimentConfig(
        experiment_type=exp_type,
        config="configs/config.yaml",
        output_dir=tmp_path,
        n_seeds=n_seeds,
        benchmark_dir="data/benchmark" if exp_type == "refute" else None,
        **kwargs,
    )
    return mod.ExperimentRunner(cfg)


class TestExperimentRunnerHelpers:
    def test_seed_dir_naming(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1)
        seed = runner.seeds[0]
        assert runner.seed_dir(seed) == tmp_path / f"seed_{seed}"

    def test_result_file_pipeline(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1)
        sd = tmp_path / "seed_42"
        assert runner.result_file(sd) == sd / "results.jsonl"

    def test_result_file_refute(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, exp_type="refute")
        sd = tmp_path / "seed_42"
        assert runner.result_file(sd) == sd / "loop_results.jsonl"

    def test_is_completed_missing(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1)
        assert runner.is_completed(runner.seeds[0]) is False

    def test_is_completed_empty_file(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1)
        seed = runner.seeds[0]
        sd = runner.seed_dir(seed)
        sd.mkdir(parents=True)
        runner.result_file(sd).write_text("")
        assert runner.is_completed(seed) is False

    def test_is_completed_nonempty(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1)
        seed = runner.seeds[0]
        sd = runner.seed_dir(seed)
        sd.mkdir(parents=True)
        runner.result_file(sd).write_text('{"ok": true}\n')
        assert runner.is_completed(seed) is True

    def test_seeds_count(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=5)
        assert len(runner.seeds) == 5

    def test_seeds_deterministic(self, mod, tmp_path) -> None:
        r1 = _make_runner(mod, tmp_path, n_seeds=4, base_seed=7)
        r2 = _make_runner(mod, tmp_path, n_seeds=4, base_seed=7)
        assert r1.seeds == r2.seeds

    def test_seeds_differ_with_different_base(self, mod, tmp_path) -> None:
        r1 = _make_runner(mod, tmp_path, n_seeds=4, base_seed=0)
        r2 = _make_runner(mod, tmp_path, n_seeds=4, base_seed=1)
        assert r1.seeds != r2.seeds


# ---------------------------------------------------------------------------
# ExperimentRunner.build_command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_pipeline_command_structure(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, domains=["number_theory"])
        seed = runner.seeds[0]
        sd = runner.seed_dir(seed)
        cmd = runner.build_command(seed, sd)
        assert "run" in cmd
        assert "--config" in cmd
        assert "--seed" in cmd
        assert str(seed) in cmd
        assert str(sd) in cmd

    def test_refute_command_structure(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, exp_type="refute")
        seed = runner.seeds[0]
        sd = runner.seed_dir(seed)
        cmd = runner.build_command(seed, sd)
        assert "refute" in cmd
        assert "--benchmark-dir" in cmd
        assert "--seed" in cmd
        assert str(seed) in cmd

    def test_pipeline_includes_domains(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, domains=["inequality", "combinatorics"])
        cmd = runner.build_command(runner.seeds[0], runner.seed_dir(runner.seeds[0]))
        assert "--domains" in cmd
        assert "inequality" in cmd
        assert "combinatorics" in cmd

    def test_pipeline_includes_n_per_domain(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, n_per_domain=50)
        cmd = runner.build_command(runner.seeds[0], runner.seed_dir(runner.seeds[0]))
        assert "--n-per-domain" in cmd
        assert "50" in cmd

    def test_refute_includes_max_rounds(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, exp_type="refute", max_rounds=8)
        cmd = runner.build_command(runner.seeds[0], runner.seed_dir(runner.seeds[0]))
        assert "--max-rounds" in cmd
        assert "8" in cmd

    def test_provider_included_when_set(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, provider="openai")
        cmd = runner.build_command(runner.seeds[0], runner.seed_dir(runner.seeds[0]))
        assert "--provider" in cmd
        assert "openai" in cmd

    def test_provider_omitted_when_none(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1, provider=None)
        cmd = runner.build_command(runner.seeds[0], runner.seed_dir(runner.seeds[0]))
        assert "--provider" not in cmd

    def test_seed_value_in_command(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1)
        seed = runner.seeds[0]
        cmd = runner.build_command(seed, runner.seed_dir(seed))
        seed_idx = cmd.index("--seed")
        assert cmd[seed_idx + 1] == str(seed)


# ---------------------------------------------------------------------------
# ExperimentRunner.run_all — subprocess + aggregation behaviour
# ---------------------------------------------------------------------------


def _fake_seed_run_metrics(idx, seed, sd):
    from conjlean.multi_seed import SeedRunMetrics
    return SeedRunMetrics(
        seed=seed,
        run_index=idx,
        metrics={"end_to_end_rate": 0.5 + idx * 0.05, "filtering_rate": 0.9},
    )


class TestRunAll:
    def _setup_all_completed(self, runner) -> None:
        """Pre-create non-empty result files for every seed."""
        for seed in runner.seeds:
            sd = runner.seed_dir(seed)
            sd.mkdir(parents=True, exist_ok=True)
            runner.result_file(sd).write_text('{"dummy": true}\n')

    def test_skips_completed_seeds(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=3)
        self._setup_all_completed(runner)

        with (
            patch("subprocess.run") as mock_sub,
            patch.object(runner, "_load_run_metrics", side_effect=_fake_seed_run_metrics),
            patch.object(runner, "_save_summary"),
        ):
            runner.run_all()
            mock_sub.assert_not_called()

    def test_runs_missing_seeds(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=3)
        # No seeds completed → subprocess called 3 times

        def _fake_execute(seed, sd):
            sd.mkdir(parents=True, exist_ok=True)
            runner.result_file(sd).write_text('{"x": 1}\n')

        with (
            patch.object(runner, "_execute_seed", side_effect=_fake_execute) as mock_exec,
            patch.object(runner, "_load_run_metrics", side_effect=_fake_seed_run_metrics),
            patch.object(runner, "_save_summary"),
        ):
            runner.run_all()
            assert mock_exec.call_count == 3

    def test_partial_resume(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=4)
        # Complete first 2 seeds
        for seed in runner.seeds[:2]:
            sd = runner.seed_dir(seed)
            sd.mkdir(parents=True, exist_ok=True)
            runner.result_file(sd).write_text('{"x": 1}\n')

        executed = []

        def _fake_execute(seed, sd):
            executed.append(seed)
            sd.mkdir(parents=True, exist_ok=True)
            runner.result_file(sd).write_text('{"x": 1}\n')

        with (
            patch.object(runner, "_execute_seed", side_effect=_fake_execute),
            patch.object(runner, "_load_run_metrics", side_effect=_fake_seed_run_metrics),
            patch.object(runner, "_save_summary"),
        ):
            runner.run_all()

        assert executed == runner.seeds[2:]

    def test_returns_aggregated_metrics(self, mod, tmp_path) -> None:
        from conjlean.multi_seed import AggregatedMetrics

        runner = _make_runner(mod, tmp_path, n_seeds=3)
        self._setup_all_completed(runner)

        with (
            patch.object(runner, "_load_run_metrics", side_effect=_fake_seed_run_metrics),
            patch.object(runner, "_save_summary"),
        ):
            result = runner.run_all()

        assert isinstance(result, AggregatedMetrics)
        assert result.n_seeds == 3

    def test_save_summary_called(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=2)
        self._setup_all_completed(runner)

        with (
            patch.object(runner, "_load_run_metrics", side_effect=_fake_seed_run_metrics),
            patch.object(runner, "_save_summary") as mock_save,
        ):
            runner.run_all()
            mock_save.assert_called_once()

    def test_seed_dir_created_before_execute(self, mod, tmp_path) -> None:
        runner = _make_runner(mod, tmp_path, n_seeds=1)
        seed = runner.seeds[0]
        expected_dir = runner.seed_dir(seed)

        created_dirs: list[Path] = []

        def _fake_sub(cmd, check=False, **kw):
            runner.result_file(expected_dir).write_text('{"x": 1}\n')
            return subprocess.CompletedProcess(cmd, 0)

        with (
            patch("subprocess.run", side_effect=_fake_sub),
            patch.object(runner, "_load_run_metrics", side_effect=_fake_seed_run_metrics),
            patch.object(runner, "_save_summary"),
        ):
            runner.run_all()

        assert expected_dir.exists()

    def test_aggregate_metrics_correct_mean(self, mod, tmp_path) -> None:
        from conjlean.multi_seed import SeedRunMetrics

        runner = _make_runner(mod, tmp_path, n_seeds=3)
        self._setup_all_completed(runner)

        metrics_seq = [
            SeedRunMetrics(seed=runner.seeds[i], run_index=i, metrics={"rate": v})
            for i, v in enumerate([0.6, 0.7, 0.8])
        ]

        with (
            patch.object(runner, "_load_run_metrics", side_effect=lambda i, s, sd: metrics_seq[i]),
            patch.object(runner, "_save_summary"),
        ):
            agg = runner.run_all()

        assert agg.mean["rate"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _save_summary — integration with MultiSeedAggregator
# ---------------------------------------------------------------------------


class TestSaveSummary:
    def test_creates_csv_and_markdown(self, mod, tmp_path) -> None:
        from conjlean.multi_seed import MultiSeedAggregator, SeedRunMetrics

        runner = _make_runner(mod, tmp_path, n_seeds=2)
        runs = [
            SeedRunMetrics(seed=runner.seeds[i], run_index=i, metrics={"f1": 0.8 + i * 0.05})
            for i in range(2)
        ]
        agg = MultiSeedAggregator().aggregate(runs)
        runner._save_summary(agg)

        expected_base = tmp_path / "multi_seed_pipeline"
        assert (tmp_path / "multi_seed_pipeline_per_seed.csv").exists()
        assert (tmp_path / "multi_seed_pipeline_aggregate.csv").exists()
        assert (tmp_path / "multi_seed_pipeline_summary.md").exists()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_missing_type_exits(self, mod, tmp_path) -> None:
        with pytest.raises(SystemExit):
            mod.main(["--config", "c.yaml", "--output-dir", str(tmp_path)])

    def test_refute_without_benchmark_exits(self, mod, tmp_path) -> None:
        result = mod.main([
            "--type", "refute",
            "--config", "c.yaml",
            "--output-dir", str(tmp_path),
            "--n-seeds", "1",
        ])
        assert result == 1

    def test_pipeline_defaults(self, mod, tmp_path) -> None:
        calls = []

        def fake_run_all(self_inner):
            calls.append(self_inner.config)
            from conjlean.multi_seed import MultiSeedAggregator, SeedRunMetrics, AggregatedMetrics
            runs = [SeedRunMetrics(seed=1, run_index=0, metrics={"r": 0.5})]
            return MultiSeedAggregator().aggregate(runs)

        with patch.object(mod.ExperimentRunner, "run_all", fake_run_all):
            result = mod.main([
                "--type", "pipeline",
                "--config", "c.yaml",
                "--output-dir", str(tmp_path),
                "--n-seeds", "2",
            ])

        assert result == 0
        assert calls[0].n_seeds == 2
        assert calls[0].experiment_type == "pipeline"

    def test_base_seed_passed_through(self, mod, tmp_path) -> None:
        calls = []

        def fake_run_all(self_inner):
            calls.append(self_inner.config)
            from conjlean.multi_seed import MultiSeedAggregator, SeedRunMetrics
            runs = [SeedRunMetrics(seed=1, run_index=0, metrics={"r": 0.5})]
            return MultiSeedAggregator().aggregate(runs)

        with patch.object(mod.ExperimentRunner, "run_all", fake_run_all):
            mod.main([
                "--type", "pipeline",
                "--config", "c.yaml",
                "--output-dir", str(tmp_path),
                "--n-seeds", "3",
                "--base-seed", "99",
            ])

        assert calls[0].base_seed == 99
