"""
Shared pytest fixtures for the ConjLean test suite.

Provides canonical test objects (conjectures, proof results, mocked clients)
that are reused across all test modules to avoid duplication and ensure
consistent baseline data shapes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from conjlean.config import ConjLeanConfig
from conjlean.schemas import (
    Conjecture,
    Domain,
    FilterResult,
    FilterStatus,
    FormalizedConjecture,
    FormalizationStatus,
    PipelineResult,
    PipelineStatus,
    ProofAttempt,
    ProofLayer,
    ProofResult,
    ProofStatus,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"

# ---------------------------------------------------------------------------
# Conjecture fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_conjecture() -> "type[_ConjectureFactory]":
    """
    Return a factory that produces a Conjecture for a given domain.

    Usage::

        def test_foo(sample_conjecture):
            c = sample_conjecture(Domain.NUMBER_THEORY)
    """

    class _ConjectureFactory:
        """Factory for sample Conjecture objects."""

        def __new__(cls, domain: Domain = Domain.NUMBER_THEORY) -> Conjecture:  # type: ignore[misc]
            return Conjecture(
                id=f"test_{domain.value}_001",
                domain=domain,
                nl_statement=_default_nl(domain),
                variables=["n"],
                source="test",
                timestamp="2026-04-15T00:00:00Z",
                metadata={"test": True},
            )

    return _ConjectureFactory


def _default_nl(domain: Domain) -> str:
    """Return a representative natural-language statement for a domain."""
    stmts = {
        Domain.NUMBER_THEORY: "2 divides n*(n+1) for all natural numbers n",
        Domain.INEQUALITY: "a^2 + b^2 >= 2*a*b for all positive reals a, b",
        Domain.COMBINATORICS: "C(2*n, n) >= 1 for n < 20",
    }
    return stmts[domain]


@pytest.fixture()
def nt_conjecture() -> Conjecture:
    """A number-theory Conjecture ready for use in tests."""
    return Conjecture(
        id="nt_test_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="2 divides n*(n+1) for all natural numbers n",
        variables=["n"],
        source="test",
    )


@pytest.fixture()
def ineq_conjecture() -> Conjecture:
    """An inequality Conjecture ready for use in tests."""
    return Conjecture(
        id="ineq_test_001",
        domain=Domain.INEQUALITY,
        nl_statement="a^2 + b^2 >= 2*a*b for all positive reals a, b",
        variables=["a", "b"],
        source="test",
    )


# ---------------------------------------------------------------------------
# FormalizedConjecture fixture
# ---------------------------------------------------------------------------

_LEAN_THEOREM = (
    "import Mathlib\n\n"
    "theorem consecutive_prod_div_two (n : ℕ) : 2 ∣ n * (n + 1) := by\n"
    "  sorry"
)


@pytest.fixture()
def sample_formalized(nt_conjecture: Conjecture) -> "type[_FormalizedFactory]":
    """
    Return a factory that builds a FormalizedConjecture with custom lean_code.

    Usage::

        def test_foo(sample_formalized):
            fc = sample_formalized("import Mathlib\\n\\ntheorem foo : True := by\\n  sorry")
    """

    class _FormalizedFactory:
        """Factory for sample FormalizedConjecture objects."""

        def __new__(cls, lean_code: str = _LEAN_THEOREM) -> FormalizedConjecture:  # type: ignore[misc]
            return FormalizedConjecture(
                conjecture=nt_conjecture,
                lean_code=lean_code,
                status=FormalizationStatus.TYPECHECKS,
                retries=0,
                error_history=[],
            )

    return _FormalizedFactory


@pytest.fixture()
def formalized_conjecture(nt_conjecture: Conjecture) -> FormalizedConjecture:
    """A FormalizedConjecture with TYPECHECKS status."""
    return FormalizedConjecture(
        conjecture=nt_conjecture,
        lean_code=_LEAN_THEOREM,
        status=FormalizationStatus.TYPECHECKS,
        retries=0,
        error_history=[],
    )


# ---------------------------------------------------------------------------
# ProofResult fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_proof_result(formalized_conjecture: FormalizedConjecture) -> "type[_ProofFactory]":
    """
    Return a factory that builds a ProofResult with a given ProofStatus.

    Usage::

        def test_foo(sample_proof_result):
            pr = sample_proof_result(ProofStatus.PROVED)
    """

    class _ProofFactory:
        """Factory for sample ProofResult objects."""

        def __new__(cls, status: ProofStatus = ProofStatus.PROVED) -> ProofResult:  # type: ignore[misc]
            return ProofResult(
                formalized=formalized_conjecture,
                status=status,
                proof=_LEAN_THEOREM.replace("sorry", "omega") if status == ProofStatus.PROVED else None,
                layer=ProofLayer.LAYER0_AUTO if status == ProofStatus.PROVED else None,
                attempts=[
                    ProofAttempt(
                        tactic="omega",
                        success=(status == ProofStatus.PROVED),
                        layer=ProofLayer.LAYER0_AUTO,
                    )
                ],
                duration_seconds=0.42,
            )

    return _ProofFactory


# ---------------------------------------------------------------------------
# PipelineResult fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_pipeline_result(
    nt_conjecture: Conjecture,
    formalized_conjecture: FormalizedConjecture,
) -> "type[_PipelineFactory]":
    """
    Return a factory that builds a PipelineResult with a given PipelineStatus.

    Usage::

        def test_foo(sample_pipeline_result):
            pr = sample_pipeline_result(PipelineStatus.PROVED)
    """

    class _PipelineFactory:
        """Factory for sample PipelineResult objects."""

        def __new__(cls, status: PipelineStatus = PipelineStatus.PROVED) -> PipelineResult:  # type: ignore[misc]
            return PipelineResult(
                conjecture=nt_conjecture,
                filter_result=FilterResult(
                    conjecture=nt_conjecture,
                    status=FilterStatus.SURVIVING,
                ),
                formalization=formalized_conjecture,
                proof=ProofResult(
                    formalized=formalized_conjecture,
                    status=ProofStatus.PROVED,
                    layer=ProofLayer.LAYER0_AUTO,
                )
                if status == PipelineStatus.PROVED
                else None,
                final_status=status,
            )

    return _PipelineFactory


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_client() -> MagicMock:
    """
    Return a MagicMock LLM client whose ``complete`` coroutine returns a
    canned Lean 4 theorem string.

    The ``complete_batch`` coroutine echoes the canned string for every input.
    """
    client = MagicMock()
    canned_lean = (
        "```lean\n"
        "import Mathlib\n\n"
        "theorem consecutive_prod_div_two (n : ℕ) : 2 ∣ n * (n + 1) := by\n"
        "  sorry\n"
        "```"
    )
    client.complete = AsyncMock(return_value=canned_lean)

    async def _batch(messages_list, temperature, max_tokens):  # type: ignore[no-untyped-def]
        return [canned_lean for _ in messages_list]

    client.complete_batch = AsyncMock(side_effect=_batch)
    return client


# ---------------------------------------------------------------------------
# Mock Lean harness
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_lean_harness() -> MagicMock:
    """
    Return a MagicMock LeanHarness with ``check_statement``, ``try_proof``,
    and ``verify_full_proof`` methods that return successful LeanCheckResults
    by default.
    """
    from conjlean.schemas import LeanCheckResult

    harness = MagicMock()
    success_result = LeanCheckResult(success=True, messages=[], env_id=1)
    harness.check_statement = MagicMock(return_value=success_result)
    harness.try_proof = MagicMock(return_value=success_result)
    harness.verify_full_proof = MagicMock(return_value=success_result)
    harness.is_running = True
    return harness


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_config() -> ConjLeanConfig:
    """
    Load a ConjLeanConfig from ``configs/config.yaml``.

    Falls back to defaults if the file is not found (so tests remain portable).
    """
    if _CONFIG_PATH.is_file():
        return ConjLeanConfig.from_yaml(str(_CONFIG_PATH))
    return ConjLeanConfig()
