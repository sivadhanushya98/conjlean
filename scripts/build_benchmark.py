"""
Build and validate the REFUTE 3-tier benchmark dataset.

This script constructs all three benchmark tiers using ``BenchmarkBuilder``,
optionally validates Tier 1 entries with SymPy to confirm their counterexamples
are genuine, then writes one JSONL file per tier and a combined ``all.jsonl``
to the output directory.

Usage
-----
::

    python scripts/build_benchmark.py [--validate] [--output-dir ./data/benchmark]

Arguments
---------
--validate
    Run SymPy-based numerical validation on every Tier 1 entry before saving.
    Entries that fail (i.e. the false statement holds for all n in 0..100)
    are excluded from the saved output and logged as warnings.
    Without this flag, all generated entries are saved unconditionally.

--output-dir PATH
    Directory where ``tier1.jsonl``, ``tier2.jsonl``, ``tier3.jsonl``, and
    ``all.jsonl`` will be written.  Created if it does not exist.
    Default: ``./data/benchmark`` relative to the project root.

--log-level {DEBUG,INFO,WARNING,ERROR}
    Logging verbosity.  Default: INFO.

Exit codes
----------
0 — success (all tiers saved).
1 — fatal error (see logs).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from conjlean.benchmark import BenchmarkBuilder, BenchmarkLoader
from conjlean.schemas import BenchmarkEntry, BenchmarkTier

# ---------------------------------------------------------------------------
# Module-level logger — configured in main() after CLI args are parsed so
# that the log level is respected from the very first message.
# ---------------------------------------------------------------------------

logger = logging.getLogger("build_benchmark")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark build script.

    Args:
        argv: Argument list to parse.  When ``None`` the parser reads from
            ``sys.argv[1:]``.

    Returns:
        Populated :class:`argparse.Namespace` with the following attributes:

        * ``validate`` (bool) — whether to run SymPy validation on Tier 1.
        * ``output_dir`` (Path) — destination directory for JSONL files.
        * ``log_level`` (str) — one of ``"DEBUG"``, ``"INFO"``,
          ``"WARNING"``, or ``"ERROR"``.
    """
    parser = argparse.ArgumentParser(
        prog="build_benchmark",
        description=(
            "Construct and optionally validate the REFUTE 3-tier benchmark "
            "dataset, then write per-tier JSONL files to the output directory.\n\n"
            "Tier 1 (~60-100 entries): Synthetic falsehoods produced by weakening "
            "conditions in known-true theorems, each with a verified counterexample.\n"
            "Tier 2 (~25 entries): Historical conjectures with known disproof, open, "
            "or proof status sourced from verified mathematical literature.\n"
            "Tier 3 (~10 entries): Subtle or imprecise statements probing scope "
            "ambiguity and implicit domain assumptions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fast build, no validation:\n"
            "  python scripts/build_benchmark.py\n\n"
            "  # Full build with SymPy validation:\n"
            "  python scripts/build_benchmark.py --validate\n\n"
            "  # Custom output directory with debug logging:\n"
            "  python scripts/build_benchmark.py --validate \\\n"
            "      --output-dir /tmp/refute_bench --log-level DEBUG\n"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help=(
            "Run SymPy-based numerical validation on every Tier 1 entry.  "
            "Entries whose synthetic false statement appears to hold for all "
            "n in 0..100 are excluded from the saved output and logged as "
            "warnings.  Reports a pass rate after validation completes.  "
            "Without this flag all generated entries are saved unconditionally.  "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark"),
        metavar="PATH",
        help=(
            "Directory where tier1.jsonl, tier2.jsonl, tier3.jsonl, and "
            "all.jsonl will be written.  Created automatically if absent.  "
            "(default: data/benchmark)"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        metavar="LEVEL",
        help=(
            "Logging verbosity level written to stderr.  "
            "Choose from DEBUG, INFO, WARNING, ERROR.  (default: INFO)"
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Summary table renderer
# ---------------------------------------------------------------------------


def _print_summary(
    t1_entries: list[BenchmarkEntry],
    t1_invalid: list[BenchmarkEntry],
    t2_entries: list[BenchmarkEntry],
    t3_entries: list[BenchmarkEntry],
    all_entries: list[BenchmarkEntry],
    output_dir: Path,
    elapsed_seconds: float,
    validated: bool,
) -> None:
    """Print a human-readable summary table to stdout.

    Uses ``print`` (not ``logging``) so the summary always reaches the
    terminal regardless of the configured log level.

    Columns: tier label, entry count, validation status.  Sections for
    per-domain and per-status breakdowns follow, plus a file inventory.

    Args:
        t1_entries: Saved (valid) Tier 1 entries after optional filtering.
        t1_invalid: Tier 1 entries excluded by SymPy validation.  Empty
            list when ``--validate`` was not requested.
        t2_entries: All Tier 2 entries built.
        t3_entries: All Tier 3 entries built.
        all_entries: Combined list written to ``all.jsonl``.
        output_dir: Resolved path to the output directory.
        elapsed_seconds: Wall-clock seconds elapsed during the build.
        validated: Whether SymPy validation was run on Tier 1.
    """
    loader = BenchmarkLoader()
    stats = loader.get_stats(all_entries)

    sep = "-" * 64
    wide_sep = "=" * 64

    t1_total = len(t1_entries) + len(t1_invalid)
    if validated and t1_total > 0:
        pass_rate = len(t1_entries) / t1_total * 100.0
        val_str = f"{len(t1_entries)}/{t1_total} passed ({pass_rate:.1f}%)"
    elif validated:
        val_str = "0/0 (no entries)"
    else:
        val_str = "not run (use --validate)"

    print()
    print(wide_sep)
    print("  REFUTE Benchmark Build Summary")
    print(wide_sep)
    print(f"  Output directory   : {output_dir.resolve()}")
    print(f"  Build time         : {elapsed_seconds:.2f}s")
    print(f"  Tier 1 validation  : {val_str}")
    print(sep)
    print(f"  {'Tier':<32} {'Entries':>8}  {'Notes'}")
    print(f"  {'-' * 32} {'-' * 8}  {'-' * 14}")
    print(f"  {'Tier 1  (Synthetic Falsehoods)':<32} {len(t1_entries):>8}  saved")
    if validated and t1_invalid:
        print(
            f"  {'  └─ excluded by validation':<32} {len(t1_invalid):>8}  skipped"
        )
    print(f"  {'Tier 2  (Historical)':<32} {len(t2_entries):>8}  saved")
    print(f"  {'Tier 3  (Subtle / Open)':<32} {len(t3_entries):>8}  saved")
    print(sep)
    print(f"  {'TOTAL':<32} {stats['total']:>8}")
    print(sep)

    print("  By domain:")
    for domain, count in sorted(stats["by_domain"].items()):
        print(f"    {domain:<28} {count:>6}")
    print(sep)

    print("  By ground-truth status:")
    for status, count in sorted(stats["by_status"].items()):
        print(f"    {status:<28} {count:>6}")
    print(sep)

    cx_count = stats["with_counterexample"]
    cx_frac = stats["fraction_with_counterexample"]
    print(
        f"  Entries with known counterexample: "
        f"{cx_count}/{stats['total']} ({cx_frac:.1%})"
    )
    print(sep)

    print("  Files written:")
    for fname in ("tier1.jsonl", "tier2.jsonl", "tier3.jsonl", "all.jsonl"):
        fpath = output_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024.0
            print(f"    {fname:<22} {size_kb:>8.1f} KB")
        else:
            print(f"    {fname:<22}   (not found)")
    print(wide_sep)
    print()


# ---------------------------------------------------------------------------
# Core build pipeline
# ---------------------------------------------------------------------------


def build(
    output_dir: Path,
    validate: bool,
) -> int:
    """Execute the full benchmark build pipeline.

    Orchestrates all build steps:

    1. Call ``BenchmarkBuilder.build_tier1()`` to generate synthetic Tier 1
       entries from the template library.
    2. If ``validate`` is True, iterate over each Tier 1 entry with a
       ``tqdm`` progress bar, calling ``builder.validate_tier1_entry()``
       per entry, collecting pass/fail counts and computing the pass rate.
    3. Call ``build_tier2()`` and ``build_tier3()`` for the remaining tiers.
    4. Persist ``tier1.jsonl``, ``tier2.jsonl``, ``tier3.jsonl``, and
       ``all.jsonl`` under ``output_dir`` via ``BenchmarkBuilder.save()``.
    5. Print a formatted summary table to stdout.

    Args:
        output_dir: Directory where benchmark JSONL files will be written.
            Created automatically if it does not exist.
        validate: When ``True``, apply per-entry SymPy validation to Tier 1
            and exclude entries that fail.

    Returns:
        Integer exit code: ``0`` on success.
    """
    t_start = time.perf_counter()

    logger.info(
        "Starting REFUTE benchmark build | output_dir=%s | validate=%s",
        output_dir,
        validate,
    )

    builder = BenchmarkBuilder()

    # ------------------------------------------------------------------
    # Tier 1 — Synthetic Falsehoods
    # ------------------------------------------------------------------
    logger.info("Building Tier 1 (synthetic falsehoods)...")
    t1_all: list[BenchmarkEntry] = builder.build_tier1()
    logger.info("Tier 1 raw build: %d entries", len(t1_all))

    t1_valid: list[BenchmarkEntry] = []
    t1_invalid: list[BenchmarkEntry] = []

    if validate:
        logger.info(
            "Running per-entry SymPy validation on %d Tier 1 entries...",
            len(t1_all),
        )
        progress = tqdm(
            t1_all,
            desc="Validating Tier 1",
            unit="entry",
            dynamic_ncols=True,
            postfix={"pass": 0, "fail": 0},
        )
        for entry in progress:
            passed: bool = builder.validate_tier1_entry(entry)
            if passed:
                t1_valid.append(entry)
            else:
                t1_invalid.append(entry)
                logger.warning(
                    "Tier 1 entry %s FAILED validation: '%s'",
                    entry.id,
                    entry.conjecture.nl_statement[:100],
                )
            progress.set_postfix(
                {"pass": len(t1_valid), "fail": len(t1_invalid)},
                refresh=False,
            )

        t1_total = len(t1_all)
        pass_rate = len(t1_valid) / t1_total * 100.0 if t1_total > 0 else 0.0
        print(
            f"\nTier 1 validation complete: {len(t1_valid)}/{t1_total} passed "
            f"({pass_rate:.1f}% pass rate)"
        )
        logger.info(
            "Tier 1 validation: %d valid | %d invalid | %.1f%% pass rate",
            len(t1_valid),
            len(t1_invalid),
            pass_rate,
        )
        t1_entries = t1_valid
    else:
        logger.info("Skipping Tier 1 validation (pass --validate to enable)")
        t1_entries = t1_all

    logger.info("Tier 1 final saved count: %d entries", len(t1_entries))

    # ------------------------------------------------------------------
    # Tier 2 — Historical Conjectures
    # ------------------------------------------------------------------
    logger.info("Building Tier 2 (historical conjectures)...")
    t2_entries: list[BenchmarkEntry] = builder.build_tier2()
    logger.info("Tier 2 built: %d entries", len(t2_entries))

    # ------------------------------------------------------------------
    # Tier 3 — Subtle / Open Cases
    # ------------------------------------------------------------------
    logger.info("Building Tier 3 (subtle/open statements)...")
    t3_entries: list[BenchmarkEntry] = builder.build_tier3()
    logger.info("Tier 3 built: %d entries", len(t3_entries))

    # ------------------------------------------------------------------
    # Save per-tier and combined JSONL files
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving benchmark files to %s...", output_dir.resolve())

    tier1_path = output_dir / "tier1.jsonl"
    tier2_path = output_dir / "tier2.jsonl"
    tier3_path = output_dir / "tier3.jsonl"
    all_path = output_dir / "all.jsonl"

    save_tasks: list[tuple[list[BenchmarkEntry], Path, str]] = [
        (t1_entries, tier1_path, "Tier 1"),
        (t2_entries, tier2_path, "Tier 2"),
        (t3_entries, tier3_path, "Tier 3"),
    ]
    for entries, path, label in tqdm(
        save_tasks,
        desc="Saving tier files",
        unit="file",
        dynamic_ncols=True,
    ):
        builder.save(entries, path)
        logger.info("%s saved: %d entries -> %s", label, len(entries), path)

    all_entries: list[BenchmarkEntry] = t1_entries + t2_entries + t3_entries
    builder.save(all_entries, all_path)
    logger.info(
        "Combined all.jsonl saved: %d entries -> %s",
        len(all_entries),
        all_path,
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    _print_summary(
        t1_entries=t1_entries,
        t1_invalid=t1_invalid,
        t2_entries=t2_entries,
        t3_entries=t3_entries,
        all_entries=all_entries,
        output_dir=output_dir,
        elapsed_seconds=elapsed,
        validated=validate,
    )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Parse CLI arguments, configure logging, and run the build pipeline.

    This is the primary entry point whether the script is invoked directly
    (``python scripts/build_benchmark.py``) or via the installed console
    script.

    Args:
        argv: Optional list of argument strings for testing.  Defaults to
            ``sys.argv[1:]`` when ``None``.

    Raises:
        SystemExit: With exit code ``0`` on success or ``1`` on fatal error.
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    try:
        exit_code = build(
            output_dir=args.output_dir,
            validate=args.validate,
        )
    except KeyboardInterrupt:
        logger.warning("Build interrupted by user.")
        sys.exit(1)
    except OSError as exc:
        logger.exception("I/O error during benchmark build: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.exception("Runtime error during benchmark build: %s", exc)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
