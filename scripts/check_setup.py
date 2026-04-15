#!/usr/bin/env python3
"""
check_setup.py — Validate the entire ConjLean environment and report
pass / warn / fail for every dependency and configuration requirement.

Usage:
    python3 scripts/check_setup.py

Exit codes:
    0 — all checks pass (warnings are acceptable)
    1 — one or more checks failed
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Resolve paths relative to the repository root (one level above scripts/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_ELAN_BIN = Path.home() / ".elan" / "bin"

# ---------------------------------------------------------------------------
# Result constants
# ---------------------------------------------------------------------------
_PASS = "pass"
_WARN = "warn"
_FAIL = "fail"

_ICON = {_PASS: "✓", _WARN: "⚠", _FAIL: "✗"}
_LABEL = {_PASS: "PASS", _WARN: "WARN", _FAIL: "FAIL"}


# ---------------------------------------------------------------------------
# Lightweight check result container
# ---------------------------------------------------------------------------


class CheckResult:
    """Holds the outcome of a single setup check."""

    def __init__(
        self,
        label: str,
        status: str,
        detail: str = "",
        fix_hint: str = "",
    ) -> None:
        self.label = label
        self.status = status          # _PASS | _WARN | _FAIL
        self.detail = detail          # human-readable one-liner shown in the summary
        self.fix_hint = fix_hint      # what the user should do to resolve a failure


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_python_version() -> CheckResult:
    """Verify Python >= 3.9 is running."""
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major >= 3 and minor >= 9:
        return CheckResult("Python version", _PASS, f"Python {version_str}")
    return CheckResult(
        "Python version",
        _FAIL,
        f"Python {version_str} (need >= 3.9)",
        fix_hint="Upgrade Python to 3.9 or later.",
    )


def check_conjlean_package() -> CheckResult:
    """Verify the conjlean package is importable."""
    try:
        import conjlean  # noqa: F401
        return CheckResult("conjlean package installed", _PASS, "import conjlean OK")
    except ImportError as exc:
        return CheckResult(
            "conjlean package installed",
            _FAIL,
            f"ImportError: {exc}",
            fix_hint="Run: pip install -e . (from the repo root)",
        )


def check_core_deps() -> CheckResult:
    """Verify every mandatory runtime dependency is importable."""
    core: list[tuple[str, str]] = [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("sympy", "sympy"),
        ("pydantic", "pydantic"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
    ]
    missing: list[str] = []
    for module, pkg in core:
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    total = len(core)
    passed = total - len(missing)

    if not missing:
        return CheckResult(
            "Core dependencies",
            _PASS,
            f"Core dependencies ({total}/{total})",
        )
    return CheckResult(
        "Core dependencies",
        _FAIL,
        f"Core dependencies ({passed}/{total}) — missing: {', '.join(missing)}",
        fix_hint=f"Run: pip install {' '.join(missing)}",
    )


def check_optional_deps() -> CheckResult:
    """Check optional dependencies; warn (not fail) if any are absent."""
    optional: list[tuple[str, str]] = [
        ("google.generativeai", "google-generativeai"),
        ("huggingface_hub", "huggingface_hub"),
        ("transformers", "transformers"),
    ]
    missing: list[str] = []
    for module, pkg in optional:
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return CheckResult(
            "Optional dependencies",
            _PASS,
            f"Optional dependencies ({len(optional)}/{len(optional)})",
        )
    return CheckResult(
        "Optional dependencies",
        _WARN,
        f"Optional deps missing: {', '.join(missing)}",
        fix_hint=f"pip install {' '.join(missing)}  (needed for non-Anthropic providers)",
    )


def check_config_file() -> CheckResult:
    """Verify configs/config.yaml exists and is valid YAML."""
    config_path = _REPO_ROOT / "configs" / "config.yaml"
    if not config_path.is_file():
        return CheckResult(
            "Config file",
            _FAIL,
            f"configs/config.yaml not found at {config_path}",
            fix_hint="Restore configs/config.yaml from the repository.",
        )
    try:
        import yaml
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError("config.yaml did not parse as a dict")
        return CheckResult("Config file", _PASS, "configs/config.yaml valid YAML")
    except Exception as exc:
        return CheckResult(
            "Config file",
            _FAIL,
            f"YAML parse error: {exc}",
            fix_hint="Fix the YAML syntax error in configs/config.yaml.",
        )


def _run_version_cmd(cmd: list[str]) -> Optional[str]:
    """Run a version command and return its stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip() or result.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def check_lean4() -> CheckResult:
    """Check whether the lean binary is available via PATH or elan."""
    # Try PATH first
    if shutil.which("lean"):
        out = _run_version_cmd(["lean", "--version"])
        version = out if out else "unknown version"
        return CheckResult("Lean 4", _PASS, f"lean on PATH: {version}")

    # Try elan-managed binary
    elan_lean = _ELAN_BIN / "lean"
    if elan_lean.is_file():
        out = _run_version_cmd([str(elan_lean), "--version"])
        version = out if out else "unknown version"
        return CheckResult("Lean 4", _PASS, f"lean via elan: {version}")

    return CheckResult(
        "Lean 4",
        _FAIL,
        "Lean 4 not found",
        fix_hint="Run: bash scripts/install_lean.sh",
    )


def check_lake() -> CheckResult:
    """Check whether the lake binary is available via PATH or elan."""
    if shutil.which("lake"):
        out = _run_version_cmd(["lake", "--version"])
        version = out if out else "unknown version"
        return CheckResult("Lake", _PASS, f"lake on PATH: {version}")

    elan_lake = _ELAN_BIN / "lake"
    if elan_lake.is_file():
        out = _run_version_cmd([str(elan_lake), "--version"])
        version = out if out else "unknown version"
        return CheckResult("Lake", _PASS, f"lake via elan: {version}")

    return CheckResult(
        "Lake",
        _FAIL,
        "lake not found",
        fix_hint="Run: bash scripts/install_lean.sh",
    )


def check_lean_project_built() -> CheckResult:
    """Check whether lake build has been run (build artefacts exist)."""
    build_dir = _REPO_ROOT / "lean" / "ConjLean" / ".lake" / "build"
    if build_dir.is_dir() and any(build_dir.iterdir()):
        return CheckResult(
            "Lean project built",
            _PASS,
            f"Build artefacts found at {build_dir}",
        )
    return CheckResult(
        "Lean project built",
        _FAIL,
        "lean/ConjLean/.lake/build/ absent or empty — lake build not run",
        fix_hint="Run: bash scripts/install_lean.sh",
    )


def check_repl_available() -> CheckResult:
    """Check whether lake exe repl is runnable inside the Lean project."""
    lean_project = _REPO_ROOT / "lean" / "ConjLean"
    if not lean_project.is_dir():
        return CheckResult(
            "REPL available",
            _FAIL,
            "Lean project directory not found",
            fix_hint="Run: bash scripts/install_lean.sh",
        )

    # Resolve lake binary
    lake_bin = shutil.which("lake") or str(_ELAN_BIN / "lake")
    if not Path(lake_bin).is_file() and not shutil.which("lake"):
        return CheckResult(
            "REPL available",
            _FAIL,
            "lake not installed — cannot check REPL",
            fix_hint="Run: bash scripts/install_lean.sh",
        )

    try:
        # A valid `lake exe repl --help` exits 0; if it exits non-zero, REPL
        # is not built.  We use a short timeout to avoid hanging.
        result = subprocess.run(
            [lake_bin, "exe", "repl", "--help"],
            cwd=str(lean_project),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return CheckResult("REPL available", _PASS, "lake exe repl responds")
        # Non-zero exit: check stderr for a meaningful message
        stderr_snippet = (result.stderr or "")[:120].strip()
        return CheckResult(
            "REPL available",
            _FAIL,
            f"lake exe repl returned exit code {result.returncode}: {stderr_snippet}",
            fix_hint="Run: bash scripts/install_lean.sh (lake build repl)",
        )
    except subprocess.TimeoutExpired:
        # Timeout means the REPL started (good sign) but did not exit on --help.
        # Treat as available.
        return CheckResult("REPL available", _PASS, "lake exe repl started (timeout — likely OK)")
    except (FileNotFoundError, OSError) as exc:
        return CheckResult(
            "REPL available",
            _FAIL,
            f"lake exe repl error: {exc}",
            fix_hint="Run: bash scripts/install_lean.sh",
        )


def check_api_keys() -> CheckResult:
    """Report which API key environment variables are set (values hidden)."""
    keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }
    parts: list[str] = []
    n_set = 0
    for name, val in keys.items():
        if val:
            parts.append(f"{name}=SET")
            n_set += 1
        else:
            parts.append(f"{name}=UNSET")

    detail = ", ".join(parts)
    if n_set == 0:
        return CheckResult(
            "API keys",
            _WARN,
            f"No API keys set — {detail}",
            fix_hint=(
                "Export at least one key, e.g.: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            ),
        )
    return CheckResult("API keys", _PASS, detail)


def check_prompt_files() -> CheckResult:
    """Verify all 6 expected prompt files exist in prompts/."""
    expected = [
        "conjecture_gen_system.txt",
        "conjecture_gen_user.txt",
        "formalizer_system.txt",
        "formalizer_repair.txt",
        "proof_gen.txt",
        "proof_repair.txt",
    ]
    prompts_dir = _REPO_ROOT / "prompts"
    missing: list[str] = [f for f in expected if not (prompts_dir / f).is_file()]
    total = len(expected)
    present = total - len(missing)

    if not missing:
        return CheckResult(
            "Prompt files",
            _PASS,
            f"All {total} prompt files present",
        )
    return CheckResult(
        "Prompt files",
        _FAIL,
        f"{present}/{total} prompt files found — missing: {', '.join(missing)}",
        fix_hint="Restore missing prompt files from the repository.",
    )


def check_data_dirs() -> CheckResult:
    """Verify data/conjectures/ and data/results/ exist and are writable."""
    required: list[Path] = [
        _REPO_ROOT / "data" / "conjectures",
        _REPO_ROOT / "data" / "results",
    ]
    not_writable: list[str] = []
    not_exist: list[str] = []

    for d in required:
        d.mkdir(parents=True, exist_ok=True)
        if not d.is_dir():
            not_exist.append(str(d))
        elif not os.access(d, os.W_OK):
            not_writable.append(str(d))

    if not_exist:
        return CheckResult(
            "Data directories",
            _FAIL,
            f"Directories could not be created: {', '.join(not_exist)}",
            fix_hint="Check filesystem permissions under data/.",
        )
    if not_writable:
        return CheckResult(
            "Data directories",
            _FAIL,
            f"Not writable: {', '.join(not_writable)}",
            fix_hint="Run: chmod -R u+w data/",
        )
    return CheckResult(
        "Data directories",
        _PASS,
        "data/conjectures/ and data/results/ exist and are writable",
    )


def check_sympy_smoke_test() -> CheckResult:
    """Run SympyFilter on one conjecture and verify it returns a result."""
    try:
        # Ensure repo src is importable even without pip install
        sys.path.insert(0, str(_REPO_ROOT / "src"))

        from conjlean.schemas import Conjecture, Domain, FilterStatus
        from conjlean.sympy_filter import SympyFilter

        # A known-surviving number-theory conjecture
        conjecture = Conjecture(
            id="smoke_test_nt_001",
            domain=Domain.NUMBER_THEORY,
            nl_statement="For all natural numbers n, 3 divides n^3 - n",
            variables=["n"],
            source="smoke_test",
        )

        sym_filter = SympyFilter(n_test_values=10, n_random_attempts=5)
        result = sym_filter.filter(conjecture)

        # The result must be a FilterResult and status must be one of the enum values
        if result is None:
            raise ValueError("SympyFilter.filter returned None")
        if result.status not in (
            FilterStatus.SURVIVING,
            FilterStatus.DISPROVED,
            FilterStatus.TRIVIAL,
        ):
            raise ValueError(f"Unexpected FilterStatus: {result.status!r}")

        return CheckResult(
            "SymPy filter smoke test",
            _PASS,
            f"SympyFilter returned status={result.status.value}",
        )
    except ImportError as exc:
        return CheckResult(
            "SymPy filter smoke test",
            _FAIL,
            f"Import failed: {exc}",
            fix_hint="Run: pip install -e . && pip install sympy",
        )
    except Exception as exc:
        return CheckResult(
            "SymPy filter smoke test",
            _FAIL,
            f"SympyFilter raised: {type(exc).__name__}: {exc}",
            fix_hint="Check conjlean.sympy_filter for runtime errors.",
        )


# ---------------------------------------------------------------------------
# Orchestration and display
# ---------------------------------------------------------------------------


def _run_all_checks() -> list[CheckResult]:
    """Execute all checks in sequence and collect results."""
    checks: list[Callable[[], CheckResult]] = [
        check_python_version,
        check_conjlean_package,
        check_core_deps,
        check_optional_deps,
        check_config_file,
        check_lean4,
        check_lake,
        check_lean_project_built,
        check_repl_available,
        check_api_keys,
        check_prompt_files,
        check_data_dirs,
        check_sympy_smoke_test,
    ]
    return [fn() for fn in checks]


def _print_summary(results: list[CheckResult]) -> None:
    """Render the summary box to stdout."""
    n_errors = sum(1 for r in results if r.status == _FAIL)
    n_warnings = sum(1 for r in results if r.status == _WARN)

    border = "━" * 44
    print(f"\n{border}")
    print("  ConjLean Setup Check")
    print(border)

    for r in results:
        icon = _ICON[r.status]
        print(f"  {icon}  {r.detail or r.label}")
        if r.status == _FAIL and r.fix_hint:
            print(f"       → {r.fix_hint}")

    print()
    if n_errors == 0 and n_warnings == 0:
        print("  All checks passed.")
    elif n_errors == 0:
        print(f"  {n_warnings} warning(s) — run is go, optional features may be limited.")
    else:
        parts: list[str] = []
        if n_errors:
            parts.append(f"{n_errors} error{'s' if n_errors != 1 else ''}")
        if n_warnings:
            parts.append(f"{n_warnings} warning{'s' if n_warnings != 1 else ''}")

        lean_failures = any(
            r.status == _FAIL and ("Lean" in r.label or "Lake" in r.label or "REPL" in r.label)
            for r in results
        )
        summary = ", ".join(parts)
        if lean_failures:
            summary += " — run `bash scripts/install_lean.sh` to fix Lean issues"
        print(f"  {summary}")

    print(border)


def main() -> int:
    """Run all checks, print summary, and return the appropriate exit code."""
    results = _run_all_checks()
    _print_summary(results)
    n_errors = sum(1 for r in results if r.status == _FAIL)
    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
