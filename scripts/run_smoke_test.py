#!/usr/bin/env python3
"""
run_smoke_test.py — End-to-end smoke test for the ConjLean pipeline.

Exercises every pipeline component using mock LLM and mock Lean backends so
that no API keys, network access, or Lean installation are required.  This
is the "can I trust the plumbing works" test.

Usage:
    python3 scripts/run_smoke_test.py

Exit codes:
    0 — PASS
    1 — FAIL
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure the repo src/ directory is on sys.path for editable-install fallback
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Logging — INFO level, clean format
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")


# ---------------------------------------------------------------------------
# Import all pipeline components
# ---------------------------------------------------------------------------
def _import_pipeline() -> bool:
    """
    Attempt to import the full pipeline and inject names into module globals.

    Returns True on success, False if any ImportError is raised.  All
    conjlean types referenced by the rest of this module are injected here so
    that the import happens once and type annotations in function signatures
    remain valid strings (i.e., forward references resolved at runtime).
    """
    try:
        from conjlean.config import ConjLeanConfig as _ConjLeanConfig
        from conjlean.schemas import (
            Conjecture as _Conjecture,
            Domain as _Domain,
            FilterResult as _FilterResult,
            FilterStatus as _FilterStatus,
            FormalizedConjecture as _FormalizedConjecture,
            FormalizationStatus as _FormalizationStatus,
            LeanCheckResult as _LeanCheckResult,
            PipelineResult as _PipelineResult,
            PipelineStatus as _PipelineStatus,
            ProofResult as _ProofResult,
            ProofStatus as _ProofStatus,
            ProofLayer as _ProofLayer,
        )
        from conjlean.pipeline import ConjLeanPipeline as _ConjLeanPipeline

        # Inject into module globals so every function below can reference
        # these types without per-function imports.
        _g = globals()
        _g["ConjLeanConfig"] = _ConjLeanConfig
        _g["Conjecture"] = _Conjecture
        _g["Domain"] = _Domain
        _g["FilterResult"] = _FilterResult
        _g["FilterStatus"] = _FilterStatus
        _g["FormalizedConjecture"] = _FormalizedConjecture
        _g["FormalizationStatus"] = _FormalizationStatus
        _g["LeanCheckResult"] = _LeanCheckResult
        _g["PipelineResult"] = _PipelineResult
        _g["PipelineStatus"] = _PipelineStatus
        _g["ProofResult"] = _ProofResult
        _g["ProofStatus"] = _ProofStatus
        _g["ProofLayer"] = _ProofLayer
        _g["ConjLeanPipeline"] = _ConjLeanPipeline
        return True
    except ImportError as exc:
        logger.error("Failed to import conjlean: %s", exc)
        logger.error("Run: pip install -e . (from repo root) to install the package.")
        return False


# ---------------------------------------------------------------------------
# Hardcoded conjecture data (used by mock generator)
# ---------------------------------------------------------------------------

_MOCK_CONJECTURES_DATA = [
    {
        "statement": "For all natural numbers n, 6 divides n*(n+1)*(n+2)",
        "variables": ["n"],
        "domain": "number_theory",
        "id": "smoke_nt_001",
    },
    {
        "statement": "For all positive reals a, b: (a + b) / 2 >= (a * b) ** 0.5",
        "variables": ["a", "b"],
        "domain": "inequality",
        "id": "smoke_ineq_001",
    },
    {
        "statement": "For all natural numbers n with n < 10, C(2*n, n) >= 1",
        "variables": ["n"],
        "domain": "combinatorics",
        "id": "smoke_comb_001",
    },
]

# Lean 4 theorem statement returned by the mock formalizer
_MOCK_LEAN_STATEMENT = (
    "import Mathlib\n\n"
    "theorem mock_conjecture (n : ℕ) : 6 ∣ n * (n + 1) * (n + 2) := by\n"
    "  sorry"
)


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """
    Mock LLM client that returns deterministic, hardcoded responses.

    For conjecture generation requests the response is a JSON-per-line list
    of three number-theory conjectures.  For formalization and proof
    generation requests a fixed Lean 4 snippet is returned.
    """

    def __init__(self) -> None:
        self._call_count: int = 0

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return a hardcoded response based on the detected call type."""
        self._call_count += 1
        content = " ".join(m.get("content", "") for m in messages).lower()

        if "conjecture" in content and "generate" in content:
            return self._conjecture_response()
        if "lean" in content or "theorem" in content or "formalize" in content:
            return self._formalization_response()
        if "tactic" in content or "proof" in content or "omega" in content:
            return self._proof_response()

        # Fallback: return a proof response (safe default for proof-search layer 3)
        return self._proof_response()

    async def complete_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """Complete a batch by calling complete() for each message list."""
        results = []
        for msgs in messages_list:
            result = await self.complete(msgs, temperature, max_tokens)
            results.append(result)
        return results

    @staticmethod
    def _conjecture_response() -> str:
        """Three hardcoded number-theory conjectures in JSON-per-line format."""
        return (
            '{"statement": "For all natural numbers n, 6 divides n*(n+1)*(n+2)", '
            '"variables": ["n"], "difficulty": "easy"}\n'
            '{"statement": "For all natural numbers n, n^2 + n is even", '
            '"variables": ["n"], "difficulty": "easy"}\n'
            '{"statement": "For all natural numbers n, 3 divides n^3 - n", '
            '"variables": ["n"], "difficulty": "easy"}'
        )

    @staticmethod
    def _formalization_response() -> str:
        """A valid Lean 4 theorem statement wrapped in a code fence."""
        return (
            "```lean\n"
            + _MOCK_LEAN_STATEMENT
            + "\n```"
        )

    @staticmethod
    def _proof_response() -> str:
        """The omega tactic as the proof body."""
        return "omega"


# ---------------------------------------------------------------------------
# Mock Lean harness
# ---------------------------------------------------------------------------


class MockLeanHarness:
    """
    Mock Lean harness that unconditionally succeeds on all verification calls.

    Tracks call counts for each method so the smoke test can assert the
    pipeline exercised every verification path.
    """

    def __init__(self) -> None:
        self.check_statement_calls: int = 0
        self.try_proof_calls: int = 0
        self.verify_full_proof_calls: int = 0

    def check_statement(self, theorem_code: str) -> "LeanCheckResult":
        """Always succeeds — simulates a Lean statement that typechecks."""
        self.check_statement_calls += 1
        logger.debug("MockLeanHarness.check_statement called (call #%d)", self.check_statement_calls)
        return LeanCheckResult(success=True, messages=[], env_id=1)

    def try_proof(self, statement_code: str, tactic_body: str) -> "LeanCheckResult":
        """Succeeds only for 'omega' to simulate layer 0 closing the proof."""
        self.try_proof_calls += 1
        logger.debug(
            "MockLeanHarness.try_proof | tactic=%r (call #%d)",
            tactic_body[:40],
            self.try_proof_calls,
        )
        # Simulate that 'omega' closes the goal; all other tactics fail
        if "omega" in tactic_body.lower():
            return LeanCheckResult(success=True, messages=[], env_id=2)
        return LeanCheckResult(
            success=False,
            messages=[{"severity": "error", "data": f"tactic '{tactic_body}' did not close the goal"}],
            env_id=1,
        )

    def verify_full_proof(self, full_lean_code: str) -> "LeanCheckResult":
        """Always succeeds — simulates a verified sorry-free proof."""
        self.verify_full_proof_calls += 1
        logger.debug(
            "MockLeanHarness.verify_full_proof called (call #%d)",
            self.verify_full_proof_calls,
        )
        return LeanCheckResult(success=True, messages=[], env_id=3)

    @property
    def is_running(self) -> bool:
        """Always report as running — no subprocess management in mock."""
        return True


# ---------------------------------------------------------------------------
# Mock SymPy filter — all conjectures survive (no subprocess needed)
# ---------------------------------------------------------------------------


class MockSympyFilter:
    """
    Mock SymPy filter that classifies every conjecture as SURVIVING.

    Avoids spawning subprocesses so the smoke test is fully self-contained.
    """

    def filter(self, conjecture: "Conjecture") -> "FilterResult":
        """Return SURVIVING for every conjecture."""
        logger.debug("MockSympyFilter.filter | id=%s", conjecture.id)
        return FilterResult(
            conjecture=conjecture,
            status=FilterStatus.SURVIVING,
            counterexample=None,
            numerical_evidence={},
        )


# ---------------------------------------------------------------------------
# Mock ConjectureGenerator
# ---------------------------------------------------------------------------


class MockConjectureGenerator:
    """
    Returns three hardcoded conjectures, one per call domain, regardless of
    the requested domain or count.
    """

    async def generate(self, domain: "Domain", n: int) -> list["Conjecture"]:
        """Return n mock conjectures for the given domain."""
        logger.debug("MockConjectureGenerator.generate | domain=%s n=%d", domain.value, n)
        conjectures: list["Conjecture"] = []
        for i, data in enumerate(_MOCK_CONJECTURES_DATA[:n], start=1):
            conjectures.append(
                Conjecture(
                    id=f"{domain.value}_smoke_{i:03d}",
                    domain=domain,
                    nl_statement=data["statement"],
                    variables=data["variables"],
                    source="mock",
                    timestamp="2026-04-15T00:00:00+00:00",
                    metadata={"difficulty": "easy"},
                )
            )
        return conjectures


# ---------------------------------------------------------------------------
# Mock Formalizer
# ---------------------------------------------------------------------------


class MockFormalizer:
    """
    Returns the fixed Lean 4 theorem snippet for every conjecture.

    Uses the mock harness to type-check so the formalization stage exercises
    the real FormalizedConjecture dataclass creation path.
    """

    def __init__(self, harness: MockLeanHarness) -> None:
        self._harness = harness

    async def formalize(self, conjecture: "Conjecture") -> "FormalizedConjecture":
        """Return a successfully type-checked FormalizedConjecture."""
        logger.debug("MockFormalizer.formalize | id=%s", conjecture.id)
        lean_code = _MOCK_LEAN_STATEMENT
        check = self._harness.check_statement(lean_code)
        status = (
            FormalizationStatus.TYPECHECKS
            if check.success
            else FormalizationStatus.UNFORMALIZABLE
        )
        return FormalizedConjecture(
            conjecture=conjecture,
            lean_code=lean_code,
            status=status,
            retries=0,
            error_history=[],
        )


# ---------------------------------------------------------------------------
# Mock ProofSearcher — exercises omega on layer 0 path
# ---------------------------------------------------------------------------


class MockProofSearcher:
    """
    Exercises the proof-search pipeline using the mock harness.

    Tries 'omega' first (which the mock harness accepts), so at least one
    result per run will have status=PROVED via layer 0.
    """

    def __init__(self, harness: MockLeanHarness) -> None:
        self._harness = harness

    async def search(self, formalized: "FormalizedConjecture") -> "ProofResult":
        """Attempt omega via the mock harness; return ProofResult."""
        from conjlean.schemas import ProofAttempt  # local to avoid forward-ref issues

        logger.debug("MockProofSearcher.search | id=%s", formalized.conjecture.id)
        tactic = "omega"
        lean_result = self._harness.try_proof(formalized.lean_code, tactic)

        if lean_result.success:
            # Confirm with verify_full_proof
            full_code = formalized.lean_code.replace("sorry", tactic, 1)
            verify_result = self._harness.verify_full_proof(full_code)
            if verify_result.success:
                return ProofResult(
                    formalized=formalized,
                    status=ProofStatus.PROVED,
                    proof=full_code,
                    layer=ProofLayer.LAYER0_AUTO,
                    attempts=[
                        ProofAttempt(
                            tactic=tactic,
                            success=True,
                            error=None,
                            layer=ProofLayer.LAYER0_AUTO,
                        )
                    ],
                    duration_seconds=0.001,
                )

        return ProofResult(
            formalized=formalized,
            status=ProofStatus.OPEN,
            proof=None,
            layer=None,
            attempts=[
                ProofAttempt(
                    tactic=tactic,
                    success=False,
                    error="mock: tactic did not close the goal",
                    layer=ProofLayer.LAYER0_AUTO,
                )
            ],
            duration_seconds=0.001,
        )


# ---------------------------------------------------------------------------
# Smoke test runner
# ---------------------------------------------------------------------------


async def _run_pipeline(save_dir: Path) -> list:
    """Instantiate all mocks and run the ConjLeanPipeline.run() coroutine."""
    # Build a minimal valid config pointing at the temp save dir
    config = ConjLeanConfig(
        provider="anthropic",
        output={"save_dir": str(save_dir)},
    )

    mock_llm = MockLLMClient()
    mock_harness = MockLeanHarness()
    mock_filter = MockSympyFilter()
    mock_generator = MockConjectureGenerator()
    mock_formalizer = MockFormalizer(mock_harness)
    mock_proof_searcher = MockProofSearcher(mock_harness)

    pipeline = ConjLeanPipeline(
        client=mock_llm,
        harness=mock_harness,
        config=config,
        generator=mock_generator,
        sym_filter=mock_filter,
        formalizer=mock_formalizer,
        proof_searcher=mock_proof_searcher,
    )

    logger.info("Running pipeline with 3 mock conjectures across 1 domain …")
    results = await pipeline.run(
        domains=[Domain.NUMBER_THEORY],
        n_per_domain=3,
    )
    return results, mock_harness


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert(condition: bool, message: str) -> None:
    """Raise AssertionError with a clear message if condition is False."""
    if not condition:
        raise AssertionError(message)


def _print_trace(results: list, harness: "MockLeanHarness") -> None:
    """Print a detailed trace of what happened in the pipeline run."""
    print("\n" + "─" * 60)
    print("  Pipeline Trace")
    print("─" * 60)
    print(f"  Conjectures processed : {len(results)}")
    print(f"  Harness.check_statement calls : {harness.check_statement_calls}")
    print(f"  Harness.try_proof calls       : {harness.try_proof_calls}")
    print(f"  Harness.verify_full_proof calls: {harness.verify_full_proof_calls}")
    print()

    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.final_status.value] = status_counts.get(r.final_status.value, 0) + 1

    for status, count in sorted(status_counts.items()):
        print(f"  {status:<20} : {count}")

    print()
    for i, r in enumerate(results, start=1):
        proved_marker = " ✓ PROVED" if r.final_status == PipelineStatus.PROVED else ""
        print(f"  [{i}] {r.conjecture.id:<30} → {r.final_status.value}{proved_marker}")
        if r.proof and r.proof.proof:
            tactic = "(no proof object)"
            if hasattr(r.proof, "proof") and r.proof.proof:
                tactic = r.proof.proof.split("by")[-1].strip()[:40]
            print(f"       proof tactic: {tactic}")

    print("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Execute the smoke test. Return 0 on PASS, 1 on FAIL."""
    print("\n" + "━" * 60)
    print("  ConjLean Smoke Test")
    print("  (mock LLM + mock Lean — no API keys or Lean required)")
    print("━" * 60 + "\n")

    # ── Import check ─────────────────────────────────────────────────────
    print("Step 1/4  Importing pipeline components …")
    if not _import_pipeline():
        print("\nFAIL — pipeline import failed (see errors above)")
        return 1
    print("          OK\n")

    # ── Run pipeline ─────────────────────────────────────────────────────
    print("Step 2/4  Running pipeline with mocks …")
    results: Optional[list] = None
    harness: Optional["MockLeanHarness"] = None

    with tempfile.TemporaryDirectory(prefix="conjlean_smoke_") as tmp_dir:
        try:
            results, harness = asyncio.run(_run_pipeline(Path(tmp_dir)))
        except Exception as exc:
            print(f"\nFAIL — pipeline raised an exception:")
            traceback.print_exc()
            return 1
    print("          OK\n")

    # ── Print trace ───────────────────────────────────────────────────────
    print("Step 3/4  Pipeline trace:")
    _print_trace(results, harness)
    print()

    # ── Assertions ────────────────────────────────────────────────────────
    print("Step 4/4  Running assertions …")
    assertion_failures: list[str] = []

    def _soft_assert(condition: bool, message: str) -> None:
        if not condition:
            assertion_failures.append(message)
            print(f"  ✗  FAIL: {message}")
        else:
            print(f"  ✓  {message}")

    _soft_assert(
        results is not None and len(results) == 3,
        f"Expected 3 PipelineResult objects, got {len(results) if results else 0}",
    )

    n_proved = sum(1 for r in results if r.final_status == PipelineStatus.PROVED)
    _soft_assert(
        n_proved >= 1,
        f"Expected at least 1 PROVED result, got {n_proved}",
    )

    _soft_assert(
        all(isinstance(r, PipelineResult) for r in results),
        "All results are PipelineResult instances",
    )

    _soft_assert(
        all(r.filter_result is not None for r in results),
        "All results have a filter_result (filtering stage ran)",
    )

    _soft_assert(
        all(r.formalization is not None for r in results),
        "All results have a formalization (formalization stage ran)",
    )

    _soft_assert(
        all(
            r.formalization.status == FormalizationStatus.TYPECHECKS
            for r in results
        ),
        "All formalized conjectures have status=TYPECHECKS",
    )

    _soft_assert(
        harness.check_statement_calls >= 3,
        f"Harness.check_statement called >= 3 times (got {harness.check_statement_calls})",
    )

    _soft_assert(
        harness.try_proof_calls >= 1,
        f"Harness.try_proof called >= 1 time (got {harness.try_proof_calls})",
    )

    _soft_assert(
        harness.verify_full_proof_calls >= 1,
        f"Harness.verify_full_proof called >= 1 time (got {harness.verify_full_proof_calls})",
    )

    # ── Final verdict ─────────────────────────────────────────────────────
    print()
    print("━" * 60)
    if assertion_failures:
        print(f"  FAIL — {len(assertion_failures)} assertion(s) failed")
        print("━" * 60 + "\n")
        return 1

    print("  PASS — all assertions passed, pipeline plumbing is healthy")
    print("━" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
