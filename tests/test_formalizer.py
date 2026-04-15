"""
Tests for conjlean.formalizer — Formalizer class, error classification,
code extraction, and batch formalization.

All LLM calls and Lean REPL calls are mocked; no real network or subprocess
activity occurs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conjlean.schemas import (
    Conjecture,
    Domain,
    FormalizedConjecture,
    FormalizationStatus,
    LeanCheckResult,
)

# ---------------------------------------------------------------------------
# Formalizer fixture helpers
# ---------------------------------------------------------------------------

_LEAN_SUCCESS = (
    "import Mathlib\n\n"
    "theorem test_thm (n : ℕ) : 2 ∣ n * (n + 1) := by\n"
    "  sorry"
)

_CANNED_LLM_RESPONSE = f"```lean\n{_LEAN_SUCCESS}\n```"


def _make_conjecture(domain: Domain = Domain.NUMBER_THEORY) -> Conjecture:
    """Build a minimal test Conjecture."""
    return Conjecture(
        id="fml_test_001",
        domain=domain,
        nl_statement="2 divides n*(n+1) for all natural numbers n",
        variables=["n"],
    )


def _make_formalizer(
    client: MagicMock,
    harness: MagicMock,
    max_retries: int = 3,
) -> "Formalizer":  # type: ignore[name-defined]
    """Instantiate a Formalizer with mocked prompts loaded from disk."""
    from conjlean.formalizer import ConjLeanConfig, Formalizer

    cfg = ConjLeanConfig()
    cfg.pipeline.max_formalization_retries = max_retries

    # Point prompt paths to real files in the repo
    repo_root = Path(__file__).resolve().parent.parent
    system_path = str(repo_root / "prompts" / "formalizer_system.txt")
    repair_path = str(repo_root / "prompts" / "formalizer_repair.txt")

    return Formalizer(
        client=client,
        harness=harness,
        config=cfg,
        system_prompt_path=system_path,
        repair_prompt_path=repair_path,
    )


def _success_harness() -> MagicMock:
    """Return a harness that always succeeds."""
    harness = MagicMock()
    harness.check_statement = MagicMock(
        return_value=LeanCheckResult(success=True, messages=[], env_id=1)
    )
    return harness


def _fail_harness(error_msg: str = "unknown identifier 'Nat.foo'") -> MagicMock:
    """Return a harness that always returns an error."""
    harness = MagicMock()
    harness.check_statement = MagicMock(
        return_value=LeanCheckResult(
            success=False,
            messages=[{"severity": "error", "data": error_msg}],
            env_id=0,
        )
    )
    return harness


# ---------------------------------------------------------------------------
# TestFormalizerSuccess
# ---------------------------------------------------------------------------


class TestFormalizerSuccess:
    """Formalization succeeds on first or second attempt."""

    @pytest.mark.asyncio()
    async def test_formalize_success_first_try(self) -> None:
        """First LLM response + successful REPL check → TYPECHECKS with retries=0."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)

        formalizer = _make_formalizer(client, _success_harness())
        result = await formalizer.formalize(_make_conjecture())

        assert result.status is FormalizationStatus.TYPECHECKS
        assert result.retries == 0
        assert result.lean_code != ""

    @pytest.mark.asyncio()
    async def test_formalize_success_after_repair(self) -> None:
        """First attempt fails, second succeeds → TYPECHECKS with retries=1."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)

        call_count = 0
        success_result = LeanCheckResult(success=True, messages=[], env_id=1)
        fail_result = LeanCheckResult(
            success=False,
            messages=[{"severity": "error", "data": "unknown identifier 'foo'"}],
            env_id=0,
        )

        def _side_effect(code: str) -> LeanCheckResult:
            nonlocal call_count
            call_count += 1
            return fail_result if call_count == 1 else success_result

        harness = MagicMock()
        harness.check_statement = MagicMock(side_effect=_side_effect)

        formalizer = _make_formalizer(client, harness, max_retries=3)
        result = await formalizer.formalize(_make_conjecture())

        assert result.status is FormalizationStatus.TYPECHECKS
        assert result.retries >= 1

    def test_extract_lean_code_fenced_lean_block(self) -> None:
        """_extract_lean_code extracts code from a ```lean ... ``` fence."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        response = "Here is the code:\n```lean\ntheorem foo : True := by\n  trivial\n```\n"
        code = formalizer._extract_lean_code(response)
        assert "theorem foo" in code
        assert "```" not in code

    def test_extract_lean_code_generic_fence(self) -> None:
        """_extract_lean_code extracts code from a generic ``` ... ``` fence."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        response = "```\ntheorem bar : 1 = 1 := by\n  rfl\n```"
        code = formalizer._extract_lean_code(response)
        assert "theorem bar" in code

    def test_extract_lean_code_bare_import(self) -> None:
        """_extract_lean_code extracts code starting with 'import Mathlib'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        response = "import Mathlib\n\ntheorem baz : True := by\n  trivial"
        code = formalizer._extract_lean_code(response)
        assert code.startswith("import Mathlib")

    def test_extract_lean_code_empty_raises(self) -> None:
        """_extract_lean_code raises ValueError for empty responses."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        with pytest.raises(ValueError, match="empty"):
            formalizer._extract_lean_code("   ")

    def test_classify_error_unknown_identifier(self) -> None:
        """Error containing 'unknown identifier' maps to 'unknown_identifier'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._classify_error("unknown identifier 'Nat.dvd'") == "unknown_identifier"

    def test_classify_error_type_mismatch(self) -> None:
        """Error containing 'type mismatch' maps to 'type_mismatch'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._classify_error("type mismatch\nexpected Nat got Int") == "type_mismatch"

    def test_classify_error_ambiguous(self) -> None:
        """Error containing 'ambiguous' maps to 'ambiguous'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._classify_error("ambiguous function overload") == "ambiguous"

    def test_classify_error_noncomputable(self) -> None:
        """Error containing 'noncomputable' maps to 'noncomputable'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._classify_error("noncomputable definition") == "noncomputable"

    def test_classify_error_missing_instance(self) -> None:
        """Error containing 'failed to synthesize' maps to 'missing_instance'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._classify_error("failed to synthesize instance [Ring R]") == "missing_instance"

    def test_classify_error_universe(self) -> None:
        """Error containing 'universe level' maps to 'universe'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._classify_error("universe level mismatch") == "universe"

    def test_classify_error_other(self) -> None:
        """Unrecognised errors map to 'other'."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._classify_error("something completely different happened") == "other"


# ---------------------------------------------------------------------------
# TestFormalizerGiveUp
# ---------------------------------------------------------------------------


class TestFormalizerGiveUp:
    """Formalizer gives up early or exhausts retries."""

    @pytest.mark.asyncio()
    async def test_gives_up_after_max_retries(self) -> None:
        """When all attempts fail, status=UNFORMALIZABLE."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)

        harness = _fail_harness("unknown identifier 'Foo'")
        formalizer = _make_formalizer(client, harness, max_retries=2)
        result = await formalizer.formalize(_make_conjecture())

        assert result.status is FormalizationStatus.UNFORMALIZABLE

    @pytest.mark.asyncio()
    async def test_gives_up_on_kernel_error(self) -> None:
        """'kernel type check failed' triggers immediate early exit → UNFORMALIZABLE."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)

        harness = _fail_harness("kernel type check failed in definition 'foo'")
        formalizer = _make_formalizer(client, harness, max_retries=5)
        result = await formalizer.formalize(_make_conjecture())

        assert result.status is FormalizationStatus.UNFORMALIZABLE
        # Should have stopped well before 5 retries
        assert client.complete.call_count < 5

    @pytest.mark.asyncio()
    async def test_gives_up_on_repeated_same_error(self) -> None:
        """The same error 3 times in a row triggers early exit → UNFORMALIZABLE."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)

        # type_mismatch three times in a row should trigger the consecutive-error heuristic
        harness = _fail_harness("type mismatch: expected Nat, got Int")
        formalizer = _make_formalizer(client, harness, max_retries=10)
        result = await formalizer.formalize(_make_conjecture())

        assert result.status is FormalizationStatus.UNFORMALIZABLE
        # Should not have exhausted all 10 retries
        assert client.complete.call_count <= 5

    @pytest.mark.asyncio()
    async def test_error_history_populated(self) -> None:
        """FormalizedConjecture.error_history accumulates all Lean errors."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)

        harness = _fail_harness("unknown identifier 'X'")
        formalizer = _make_formalizer(client, harness, max_retries=3)
        result = await formalizer.formalize(_make_conjecture())

        assert isinstance(result.error_history, list)
        assert len(result.error_history) >= 1
        assert all(isinstance(e, str) for e in result.error_history)

    def test_should_give_up_on_kernel_error(self) -> None:
        """_should_give_up returns True when kernel error is in history."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        history = ["kernel type check failed in foo", "some other error"]
        assert formalizer._should_give_up(history) is True

    def test_should_not_give_up_on_empty_history(self) -> None:
        """_should_give_up returns False for empty error history."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        assert formalizer._should_give_up([]) is False

    def test_should_give_up_on_three_identical_errors(self) -> None:
        """_should_give_up returns True when the same error appears 3 times in a row."""
        client = MagicMock()
        formalizer = _make_formalizer(client, MagicMock())
        history = [
            "type mismatch: expected Nat, got Int",
            "type mismatch: expected Nat, got Int",
            "type mismatch: expected Nat, got Int",
        ]
        assert formalizer._should_give_up(history) is True


# ---------------------------------------------------------------------------
# TestFormalizerBatch
# ---------------------------------------------------------------------------


class TestFormalizerBatch:
    """Batch formalization tests."""

    @pytest.mark.asyncio()
    async def test_batch_concurrent_returns_all(self) -> None:
        """formalize_batch returns exactly one result per input conjecture."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)
        harness = _success_harness()
        formalizer = _make_formalizer(client, harness)

        conjectures = [
            Conjecture(
                id=f"batch_{i}",
                domain=Domain.NUMBER_THEORY,
                nl_statement=f"conjecture {i}",
                variables=["n"],
            )
            for i in range(10)
        ]

        results = await formalizer.formalize_batch(conjectures, max_concurrent=5)
        assert len(results) == 10

    @pytest.mark.asyncio()
    async def test_batch_order_preserved(self) -> None:
        """formalize_batch preserves input ordering in the returned results."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)
        harness = _success_harness()
        formalizer = _make_formalizer(client, harness)

        conjectures = [
            Conjecture(
                id=f"order_{i}",
                domain=Domain.NUMBER_THEORY,
                nl_statement=f"stmt {i}",
                variables=["n"],
            )
            for i in range(6)
        ]
        results = await formalizer.formalize_batch(conjectures, max_concurrent=3)
        for i, result in enumerate(results):
            assert result.conjecture.id == f"order_{i}", (
                f"Position {i}: expected id='order_{i}', got '{result.conjecture.id}'"
            )

    @pytest.mark.asyncio()
    async def test_batch_respects_semaphore(self) -> None:
        """formalize_batch never exceeds max_concurrent simultaneous tasks."""
        import asyncio as _asyncio

        max_concurrent = 3
        active: list[int] = []
        peak: list[int] = [0]

        client = MagicMock()

        async def _track_complete(*args, **kwargs):  # type: ignore[no-untyped-def]
            active.append(1)
            peak[0] = max(peak[0], len(active))
            await _asyncio.sleep(0)
            active.pop()
            return _CANNED_LLM_RESPONSE

        client.complete = AsyncMock(side_effect=_track_complete)
        harness = _success_harness()
        formalizer = _make_formalizer(client, harness)

        conjectures = [
            Conjecture(
                id=f"sem_{i}",
                domain=Domain.NUMBER_THEORY,
                nl_statement=f"stmt {i}",
                variables=["n"],
            )
            for i in range(max_concurrent * 2)
        ]
        await formalizer.formalize_batch(conjectures, max_concurrent=max_concurrent)
        # Peak concurrency should not exceed the semaphore limit
        assert peak[0] <= max_concurrent

    @pytest.mark.asyncio()
    async def test_batch_empty_returns_empty(self) -> None:
        """formalize_batch on an empty list returns an empty list."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=_CANNED_LLM_RESPONSE)
        formalizer = _make_formalizer(client, _success_harness())
        results = await formalizer.formalize_batch([])
        assert results == []
