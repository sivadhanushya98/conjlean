"""
Tests for scripts/validate_benchmark.py.

All tests run in-process against real SympyFilter checkers or mocked
BenchmarkEntry objects.  No subprocess spawning; no benchmark JSONL files
required on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_benchmark import (  # noqa: E402
    EntryValidation,
    ValidationResult,
    _run_checker,
    main,
    print_report,
    validate_all,
    validate_entry,
)

from conjlean.schemas import BenchmarkEntry, BenchmarkTier, Conjecture, Domain, FilterStatus
from conjlean.sympy_filter import _SympyCheckers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_id: str = "test_001",
    nl_statement: str = "12 divides n*(n+1)*(n+2)",
    domain: Domain = Domain.NUMBER_THEORY,
    ground_truth_status: str = "false",
    ground_truth_counterexample: str | None = "n=1: 1*2*3=6, not divisible by 12",
    tier: BenchmarkTier = BenchmarkTier.TIER1_SYNTHETIC,
) -> BenchmarkEntry:
    conjecture = Conjecture(
        id=entry_id,
        domain=domain,
        nl_statement=nl_statement,
        variables=["n"],
    )
    return BenchmarkEntry(
        id=entry_id,
        conjecture=conjecture,
        tier=tier,
        ground_truth_counterexample=ground_truth_counterexample,
        ground_truth_status=ground_truth_status,
    )


def _mock_checkers(status: FilterStatus, counterexample: str = "") -> _SympyCheckers:
    """Return a _SympyCheckers whose check_* methods always return a fixed status."""
    from conjlean.sympy_filter import _CheckResult

    fixed = _CheckResult(status=status, counterexample=counterexample or None)
    checkers = MagicMock(spec=_SympyCheckers)
    checkers.check_number_theory.return_value = fixed
    checkers.check_inequality.return_value = fixed
    checkers.check_combinatorics.return_value = fixed
    return checkers


# ---------------------------------------------------------------------------
# validate_entry
# ---------------------------------------------------------------------------


class TestValidateEntryOpenTrue:
    def test_open_status_skipped(self) -> None:
        entry = _make_entry(ground_truth_status="open", ground_truth_counterexample=None)
        result = validate_entry(entry, _mock_checkers(FilterStatus.SURVIVING))
        assert result.result == ValidationResult.NO_CE_EXPECTED

    def test_true_status_skipped(self) -> None:
        entry = _make_entry(ground_truth_status="true", ground_truth_counterexample=None)
        result = validate_entry(entry, _mock_checkers(FilterStatus.SURVIVING))
        assert result.result == ValidationResult.NO_CE_EXPECTED

    def test_true_with_caveat_skipped(self) -> None:
        entry = _make_entry(
            ground_truth_status="true_with_caveat", ground_truth_counterexample=None
        )
        result = validate_entry(entry, _mock_checkers(FilterStatus.SURVIVING))
        assert result.result == ValidationResult.NO_CE_EXPECTED

    def test_no_checker_call_for_open(self) -> None:
        """Checker should never be called for non-false entries."""
        checkers = _mock_checkers(FilterStatus.SURVIVING)
        entry = _make_entry(ground_truth_status="open", ground_truth_counterexample=None)
        validate_entry(entry, checkers)
        checkers.check_number_theory.assert_not_called()


class TestValidateEntryFalse:
    def test_disproved_gives_verified(self) -> None:
        entry = _make_entry(ground_truth_status="false")
        result = validate_entry(entry, _mock_checkers(FilterStatus.DISPROVED, "n=1: 6 mod 12=6"))
        assert result.result == ValidationResult.VERIFIED
        assert "filter confirmed" in result.detail

    def test_surviving_gives_unverifiable(self) -> None:
        entry = _make_entry(ground_truth_status="false")
        result = validate_entry(entry, _mock_checkers(FilterStatus.SURVIVING))
        assert result.result == ValidationResult.UNVERIFIABLE
        assert "SURVIVING" in result.detail

    def test_trivial_gives_invalid(self) -> None:
        entry = _make_entry(ground_truth_status="false")
        result = validate_entry(entry, _mock_checkers(FilterStatus.TRIVIAL))
        assert result.result == ValidationResult.INVALID
        assert "TRIVIAL" in result.detail

    def test_unverifiable_includes_stated_ce(self) -> None:
        entry = _make_entry(
            ground_truth_status="false",
            ground_truth_counterexample="n=42: value is 123",
        )
        result = validate_entry(entry, _mock_checkers(FilterStatus.SURVIVING))
        assert "n=42" in result.detail

    def test_checker_exception_gives_unverifiable(self) -> None:
        checkers = MagicMock(spec=_SympyCheckers)
        checkers.check_number_theory.side_effect = RuntimeError("boom")
        entry = _make_entry(ground_truth_status="false")
        result = validate_entry(entry, checkers)
        assert result.result == ValidationResult.UNVERIFIABLE

    def test_metadata_fields_correct(self) -> None:
        entry = _make_entry(
            entry_id="my_id",
            ground_truth_status="false",
            tier=BenchmarkTier.TIER2_HISTORICAL,
        )
        result = validate_entry(entry, _mock_checkers(FilterStatus.DISPROVED))
        assert result.entry_id == "my_id"
        assert result.tier == "tier2_historical"
        assert result.ground_truth_status == "false"


# ---------------------------------------------------------------------------
# _run_checker — domain routing
# ---------------------------------------------------------------------------


class TestRunChecker:
    def test_routes_number_theory(self) -> None:
        entry = _make_entry(domain=Domain.NUMBER_THEORY)
        checkers = _mock_checkers(FilterStatus.DISPROVED, "ce")
        status, _ = _run_checker(entry, checkers)
        checkers.check_number_theory.assert_called_once()
        assert status == FilterStatus.DISPROVED

    def test_routes_inequality(self) -> None:
        entry = _make_entry(
            domain=Domain.INEQUALITY,
            nl_statement="a^2 + b^2 >= 2*a*b for positive reals",
        )
        checkers = _mock_checkers(FilterStatus.SURVIVING)
        status, _ = _run_checker(entry, checkers)
        checkers.check_inequality.assert_called_once()

    def test_routes_combinatorics(self) -> None:
        entry = _make_entry(
            domain=Domain.COMBINATORICS,
            nl_statement="C(2n, n) = 2 for n >= 1",
        )
        checkers = _mock_checkers(FilterStatus.SURVIVING)
        status, _ = _run_checker(entry, checkers)
        checkers.check_combinatorics.assert_called_once()

    def test_exception_returns_surviving(self) -> None:
        entry = _make_entry()
        checkers = MagicMock(spec=_SympyCheckers)
        checkers.check_number_theory.side_effect = ValueError("oops")
        status, detail = _run_checker(entry, checkers)
        assert status == FilterStatus.SURVIVING
        assert "ValueError" in detail


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------


class TestValidateAll:
    def test_returns_one_result_per_entry(self) -> None:
        entries = [_make_entry(entry_id=f"e{i}") for i in range(5)]
        checkers = _mock_checkers(FilterStatus.DISPROVED)
        results = validate_all(entries, checkers)
        assert len(results) == 5

    def test_default_checkers_created_when_none(self) -> None:
        entries = [_make_entry(ground_truth_status="open", ground_truth_counterexample=None)]
        results = validate_all(entries)
        assert len(results) == 1
        assert results[0].result == ValidationResult.NO_CE_EXPECTED

    def test_mixed_results(self) -> None:
        entries = [
            _make_entry("e1", ground_truth_status="false"),
            _make_entry("e2", ground_truth_status="open", ground_truth_counterexample=None),
        ]
        results = validate_all(entries, _mock_checkers(FilterStatus.DISPROVED))
        statuses = {v.entry_id: v.result for v in results}
        assert statuses["e1"] == ValidationResult.VERIFIED
        assert statuses["e2"] == ValidationResult.NO_CE_EXPECTED


# ---------------------------------------------------------------------------
# print_report — smoke test (just ensure it doesn't crash)
# ---------------------------------------------------------------------------


class TestPrintReport:
    def test_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        validations = [
            EntryValidation("e1", "tier1_synthetic", "number_theory", "false",
                            ValidationResult.VERIFIED, "ok"),
            EntryValidation("e2", "tier2_historical", "number_theory", "open",
                            ValidationResult.NO_CE_EXPECTED, "skipped"),
            EntryValidation("e3", "tier1_synthetic", "number_theory", "false",
                            ValidationResult.UNVERIFIABLE, "filter surviving"),
        ]
        print_report(validations)
        out = capsys.readouterr().out
        assert "VERIFIED" in out
        assert "UNVERIFIABLE" in out
        assert "NO_CE_EXPECTED" not in out  # skipped entries not in checked counts

    def test_invalid_entries_highlighted(self, capsys: pytest.CaptureFixture) -> None:
        validations = [
            EntryValidation("bad_001", "tier1_synthetic", "number_theory", "false",
                            ValidationResult.INVALID, "TRIVIAL"),
        ]
        print_report(validations)
        out = capsys.readouterr().out
        assert "INVALID" in out
        assert "bad_001" in out


# ---------------------------------------------------------------------------
# main() — integration using real benchmark data when available
# ---------------------------------------------------------------------------


class TestMain:
    def test_missing_benchmark_dir_exits_1(self, tmp_path: Path) -> None:
        rc = main(["--benchmark-dir", str(tmp_path / "nonexistent")])
        assert rc == 1

    def test_empty_benchmark_dir_exits_1(self, tmp_path: Path) -> None:
        rc = main(["--benchmark-dir", str(tmp_path)])
        assert rc == 1

    def test_no_invalid_entries_exits_0(self, tmp_path: Path) -> None:
        import dataclasses, json
        all_jsonl = tmp_path / "all.jsonl"
        entry = _make_entry(
            ground_truth_status="open",
            ground_truth_counterexample=None,
            tier=BenchmarkTier.TIER2_HISTORICAL,
        )
        all_jsonl.write_text(
            json.dumps(dataclasses.asdict(entry)) + "\n", encoding="utf-8"
        )
        with patch("validate_benchmark.BenchmarkLoader") as MockLoader:
            MockLoader.return_value.load_all.return_value = [entry]
            rc = main(["--benchmark-dir", str(tmp_path)])
        assert rc == 0

    def test_invalid_entry_exits_1(self, tmp_path: Path) -> None:
        import dataclasses, json
        all_jsonl = tmp_path / "all.jsonl"
        entry = _make_entry(ground_truth_status="false")
        all_jsonl.write_text(
            json.dumps(dataclasses.asdict(entry)) + "\n", encoding="utf-8"
        )
        with patch("validate_benchmark.BenchmarkLoader") as MockLoader, \
             patch("validate_benchmark._run_checker", return_value=(FilterStatus.TRIVIAL, "")):
            MockLoader.return_value.load_all.return_value = [entry]
            rc = main(["--benchmark-dir", str(tmp_path)])
        assert rc == 1

    def test_tier_filter_restricts_entries(self, tmp_path: Path) -> None:
        import dataclasses, json
        entries = [
            _make_entry("t1", tier=BenchmarkTier.TIER1_SYNTHETIC, ground_truth_status="open",
                        ground_truth_counterexample=None),
            _make_entry("t2", tier=BenchmarkTier.TIER2_HISTORICAL, ground_truth_status="open",
                        ground_truth_counterexample=None),
        ]
        lines = "\n".join(json.dumps(dataclasses.asdict(e)) for e in entries)
        (tmp_path / "all.jsonl").write_text(lines + "\n", encoding="utf-8")

        with patch("validate_benchmark.BenchmarkLoader") as MockLoader, \
             patch("validate_benchmark.validate_all") as mock_va:
            MockLoader.return_value.load_all.return_value = entries
            mock_va.return_value = []
            main(["--benchmark-dir", str(tmp_path), "--tier", "tier1"])
            called_entries = mock_va.call_args[0][0]
        assert all(e.tier == BenchmarkTier.TIER1_SYNTHETIC for e in called_entries)
        assert len(called_entries) == 1
