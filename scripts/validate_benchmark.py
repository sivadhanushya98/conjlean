"""
Programmatic validation of all saved REFUTE benchmark entries.

Loads the built benchmark (data/benchmark/all.jsonl by default) and runs the
in-process SympyFilter checker on every entry whose ground_truth_status is
"false", independently confirming the conjecture is genuinely falsifiable.

Classification
--------------
VERIFIED      -- SympyFilter independently finds a counterexample.
UNVERIFIABLE  -- Filter returns SURVIVING; the CE is non-constructive or
                 requires reasoning beyond SymPy's pattern library.
INVALID       -- Filter classifies the entry as TRIVIAL -- benchmark bug.
NO_CE_EXPECTED -- Entry is open / true / true_with_caveat; skip.

Exit code
---------
0 -- no INVALID entries (all "false" entries are either confirmed or
     unverifiable by the filter).
1 -- one or more INVALID entries detected (benchmark construction error).

Usage
-----
    python scripts/validate_benchmark.py
    python scripts/validate_benchmark.py --benchmark-dir data/benchmark
    python scripts/validate_benchmark.py --tier tier1
    python scripts/validate_benchmark.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from conjlean.benchmark import BenchmarkLoader
from conjlean.schemas import BenchmarkEntry, Domain
from conjlean.sympy_filter import FilterStatus, _SympyCheckers

logger = logging.getLogger("validate_benchmark")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ValidationResult(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIABLE = "UNVERIFIABLE"
    INVALID = "INVALID"
    NO_CE_EXPECTED = "NO_CE_EXPECTED"


@dataclass
class EntryValidation:
    entry_id: str
    tier: str
    domain: str
    ground_truth_status: str
    result: ValidationResult
    detail: str


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def _run_checker(
    entry: BenchmarkEntry,
    checkers: _SympyCheckers,
) -> tuple[FilterStatus, str]:
    """Dispatch to the domain-appropriate in-process checker.

    Returns (filter_status, ce_detail).  Never raises — exceptions become
    a SURVIVING result with an error note.
    """
    conjecture = entry.conjecture
    try:
        if conjecture.domain == Domain.NUMBER_THEORY:
            result = checkers.check_number_theory(conjecture)
        elif conjecture.domain == Domain.INEQUALITY:
            result = checkers.check_inequality(conjecture)
        elif conjecture.domain == Domain.COMBINATORICS:
            result = checkers.check_combinatorics(conjecture)
        else:
            return FilterStatus.SURVIVING, f"unknown domain: {conjecture.domain}"
        return result.status, result.counterexample or ""
    except Exception as exc:  # noqa: BLE001
        return FilterStatus.SURVIVING, f"checker error: {type(exc).__name__}: {exc}"


def validate_entry(
    entry: BenchmarkEntry,
    checkers: _SympyCheckers,
) -> EntryValidation:
    """Validate one benchmark entry.

    For entries with ground_truth_status in ("open", "true",
    "true_with_caveat"), validation is skipped (NO_CE_EXPECTED).
    For "false" entries, the in-process SympyFilter checker is run.
    """
    base = dict(
        entry_id=entry.id,
        tier=entry.tier.value,
        domain=str(entry.conjecture.domain),
        ground_truth_status=entry.ground_truth_status,
    )

    if entry.ground_truth_status != "false":
        return EntryValidation(
            **base,
            result=ValidationResult.NO_CE_EXPECTED,
            detail=f"status={entry.ground_truth_status!r} — no CE verification required",
        )

    filter_status, ce_detail = _run_checker(entry, checkers)

    if filter_status == FilterStatus.DISPROVED:
        return EntryValidation(
            **base,
            result=ValidationResult.VERIFIED,
            detail=f"filter confirmed: {ce_detail}" if ce_detail else "filter confirmed",
        )

    if filter_status == FilterStatus.TRIVIAL:
        return EntryValidation(
            **base,
            result=ValidationResult.INVALID,
            detail="filter classifies statement as TRIVIAL — benchmark construction error",
        )

    # FilterStatus.SURVIVING — could not independently confirm
    stated_ce = entry.ground_truth_counterexample or "(none)"
    return EntryValidation(
        **base,
        result=ValidationResult.UNVERIFIABLE,
        detail=(
            f"filter returned SURVIVING; stated CE: {stated_ce[:120]}"
            if len(stated_ce) > 120
            else f"filter returned SURVIVING; stated CE: {stated_ce}"
        ),
    )


def validate_all(
    entries: list[BenchmarkEntry],
    checkers: Optional[_SympyCheckers] = None,
) -> list[EntryValidation]:
    """Validate all entries, returning one EntryValidation per entry."""
    if checkers is None:
        checkers = _SympyCheckers(n_test_values=20, n_random_attempts=10)
    return [validate_entry(e, checkers) for e in entries]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(validations: list[EntryValidation]) -> None:
    """Print a human-readable validation report to stdout."""
    counts: dict[ValidationResult, int] = {r: 0 for r in ValidationResult}
    invalid_entries: list[EntryValidation] = []
    tier_counts: dict[str, dict[ValidationResult, int]] = {}

    for v in validations:
        counts[v.result] += 1
        tier_counts.setdefault(v.tier, {r: 0 for r in ValidationResult})
        tier_counts[v.tier][v.result] += 1
        if v.result == ValidationResult.INVALID:
            invalid_entries.append(v)

    total = len(validations)
    false_total = total - counts[ValidationResult.NO_CE_EXPECTED]

    print()
    print("=" * 68)
    print("  REFUTE Benchmark Validation Report")
    print("=" * 68)
    print(f"  Total entries       : {total}")
    print(f"  Entries checked     : {false_total}  (ground_truth_status=false)")
    print(f"  Skipped (open/true) : {counts[ValidationResult.NO_CE_EXPECTED]}")
    print()

    # Per-result summary
    col_w = 16
    print(f"  {'Result':<{col_w}}  {'Count':>6}  {'of checked':>10}")
    print(f"  {'-'*col_w}  {'-'*6}  {'-'*10}")
    for result in (
        ValidationResult.VERIFIED,
        ValidationResult.UNVERIFIABLE,
        ValidationResult.INVALID,
    ):
        pct = f"{counts[result]/false_total:.0%}" if false_total else "n/a"
        print(f"  {result.value:<{col_w}}  {counts[result]:>6}  {pct:>10}")

    # Per-tier breakdown
    print()
    print("  Per-tier breakdown:")
    tiers = sorted(tier_counts)
    for tier in tiers:
        tc = tier_counts[tier]
        tier_total = sum(tc.values())
        verified = tc[ValidationResult.VERIFIED]
        unverif = tc[ValidationResult.UNVERIFIABLE]
        invalid = tc[ValidationResult.INVALID]
        no_ce = tc[ValidationResult.NO_CE_EXPECTED]
        print(
            f"    {tier:<24}  total={tier_total}"
            f"  verified={verified}  unverifiable={unverif}"
            f"  invalid={invalid}  skipped={no_ce}"
        )

    # INVALID entries — prominent warning
    if invalid_entries:
        print()
        print("  [ERROR] INVALID entries detected — benchmark construction errors:")
        for v in invalid_entries:
            print(f"    {v.entry_id}  ({v.tier})  {v.detail}")
    else:
        print()
        print("  No INVALID entries detected.")

    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_benchmark",
        description=(
            "Independently verify every REFUTE benchmark entry whose "
            "ground_truth_status is 'false' using the in-process SympyFilter."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        metavar="PATH",
        default="data/benchmark",
        help="Directory containing all.jsonl (default: data/benchmark)",
    )
    parser.add_argument(
        "--tier",
        choices=["tier1", "tier2", "tier3", "all"],
        default="all",
        help="Which tier to validate (default: all)",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    benchmark_dir = Path(args.benchmark_dir)
    all_jsonl = benchmark_dir / "all.jsonl"

    if not all_jsonl.exists():
        print(
            f"ERROR: {all_jsonl} not found.\n"
            "Run 'make build-benchmark' first to generate the benchmark files.",
            file=sys.stderr,
        )
        return 1

    loader = BenchmarkLoader()
    all_entries = loader.load_all(benchmark_dir)

    if args.tier != "all":
        tier_map = {
            "tier1": "tier1_synthetic",
            "tier2": "tier2_historical",
            "tier3": "tier3_subtle",
        }
        target = tier_map[args.tier]
        entries = [e for e in all_entries if e.tier.value == target]
        logger.info("Filtered to %d entries for %s", len(entries), args.tier)
    else:
        entries = all_entries

    if not entries:
        print("No entries found. Is the benchmark built?", file=sys.stderr)
        return 1

    checkers = _SympyCheckers(n_test_values=20, n_random_attempts=10)
    validations = validate_all(entries, checkers)
    print_report(validations)

    invalid_count = sum(1 for v in validations if v.result == ValidationResult.INVALID)
    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
