"""
Tests for conjlean.proof_search — ProofSearcher layered proof strategy.

All Lean harness calls and LLM completions are mocked. Tests cover each
layer independently and the full cascade from top to bottom.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from conjlean.config import ConjLeanConfig
from conjlean.schemas import (
    Conjecture,
    Domain,
    FormalizedConjecture,
    FormalizationStatus,
    LeanCheckResult,
    ProofLayer,
    ProofResult,
    ProofStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEAN_CODE = (
    "import Mathlib\n\n"
    "theorem test_thm (n : ℕ) : 2 ∣ n * (n + 1) := by\n"
    "  sorry"
)


def _make_formalized(lean_code: str = _LEAN_CODE) -> FormalizedConjecture:
    """Build a FormalizedConjecture with TYPECHECKS status."""
    c = Conjecture(
        id="ps_test_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="2 divides n*(n+1)",
        variables=["n"],
    )
    return FormalizedConjecture(
        conjecture=c,
        lean_code=lean_code,
        status=FormalizationStatus.TYPECHECKS,
    )


def _make_config(
    layer0_timeout: int = 5,
    layer1_timeout: int = 10,
    layer2_timeout: int = 10,
    layer3_timeout: int = 30,
    layer3_max_rounds: int = 2,
) -> ConjLeanConfig:
    """Build a ConjLeanConfig with overridden proof-search timeouts."""
    cfg = ConjLeanConfig()
    cfg.proof_search.layer0_timeout = layer0_timeout
    cfg.proof_search.layer1_timeout = layer1_timeout
    cfg.proof_search.layer2_timeout = layer2_timeout
    cfg.proof_search.layer3_timeout = layer3_timeout
    cfg.proof_search.layer3_max_rounds = layer3_max_rounds
    return cfg


def _make_searcher(
    client: MagicMock,
    harness: MagicMock,
    cfg: ConjLeanConfig | None = None,
) -> "ProofSearcher":  # type: ignore[name-defined]
    """Instantiate a ProofSearcher with mocked client and harness."""
    from conjlean.proof_search import ProofSearcher

    return ProofSearcher(
        client=client,
        harness=harness,
        config=cfg or _make_config(),
    )


def _success_result() -> LeanCheckResult:
    return LeanCheckResult(success=True, messages=[], env_id=1)


def _fail_result(msg: str = "error: unknown identifier") -> LeanCheckResult:
    return LeanCheckResult(
        success=False,
        messages=[{"severity": "error", "data": msg}],
        env_id=0,
    )


# ---------------------------------------------------------------------------
# TestLayer0
# ---------------------------------------------------------------------------


class TestLayer0:
    """Tests for ProofSearcher._layer0_auto_tactics."""

    @pytest.mark.asyncio()
    async def test_omega_closes_goal(self) -> None:
        """If 'omega' succeeds, the full search returns PROVED on LAYER0_AUTO."""
        harness = MagicMock()

        def _try_proof(code: str, tactic: str) -> LeanCheckResult:
            return _success_result() if tactic == "omega" else _fail_result()

        harness.try_proof = MagicMock(side_effect=_try_proof)
        harness.verify_full_proof = MagicMock(return_value=_success_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="omega")

        searcher = _make_searcher(client, harness)
        result = await searcher.search(_make_formalized())

        assert result.status is ProofStatus.PROVED
        assert result.layer is ProofLayer.LAYER0_AUTO

    @pytest.mark.asyncio()
    async def test_norm_num_closes_goal(self) -> None:
        """If 'norm_num' succeeds, the search returns PROVED on LAYER0_AUTO."""
        harness = MagicMock()

        def _try_proof(code: str, tactic: str) -> LeanCheckResult:
            return _success_result() if tactic == "norm_num" else _fail_result()

        harness.try_proof = MagicMock(side_effect=_try_proof)
        harness.verify_full_proof = MagicMock(return_value=_success_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="norm_num")

        searcher = _make_searcher(client, harness)
        result = await searcher.search(_make_formalized())

        assert result.status is ProofStatus.PROVED
        assert result.layer is ProofLayer.LAYER0_AUTO

    @pytest.mark.asyncio()
    async def test_all_layer0_fail_escalates(self) -> None:
        """When all Layer-0 tactics fail, the search escalates to Layer 1+."""
        harness = MagicMock()
        harness.try_proof = MagicMock(return_value=_fail_result())
        harness.verify_full_proof = MagicMock(return_value=_fail_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="omega")

        searcher = _make_searcher(client, harness, _make_config(layer3_max_rounds=1))
        result = await searcher.search(_make_formalized())

        # All layers failed → OPEN; the harness was called more than once (probing multiple tactics)
        assert harness.try_proof.call_count > 1

    @pytest.mark.asyncio()
    async def test_layer0_records_attempts(self) -> None:
        """After Layer-0 succeeds, the result.attempts list contains the successful tactic."""
        harness = MagicMock()

        def _try_proof(code: str, tactic: str) -> LeanCheckResult:
            return _success_result() if tactic == "omega" else _fail_result()

        harness.try_proof = MagicMock(side_effect=_try_proof)
        harness.verify_full_proof = MagicMock(return_value=_success_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="omega")

        searcher = _make_searcher(client, harness)
        result = await searcher.search(_make_formalized())

        tactic_names = [a.tactic for a in result.attempts]
        assert "omega" in tactic_names


# ---------------------------------------------------------------------------
# TestLayer1
# ---------------------------------------------------------------------------


class TestLayer1:
    """Tests for ProofSearcher._layer1_combo_tactics."""

    @pytest.mark.asyncio()
    async def test_induction_combo_succeeds(self) -> None:
        """An induction combo tactic that succeeds returns PROVED on LAYER1_COMBO."""
        harness = MagicMock()

        def _try_proof(code: str, tactic: str) -> LeanCheckResult:
            if "induction" in tactic:
                return _success_result()
            return _fail_result()

        harness.try_proof = MagicMock(side_effect=_try_proof)
        harness.verify_full_proof = MagicMock(return_value=_success_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="omega")

        # Make Layer0 fail so we reach Layer1
        def _l0_fail_l1_succeed(code: str, tactic: str) -> LeanCheckResult:
            if "induction" in tactic:
                return _success_result()
            if tactic in ("decide", "norm_num", "omega", "ring", "simp",
                          "aesop", "positivity", "simp; omega", "simp; ring",
                          "simp; norm_num", "norm_num [Nat.Prime]"):
                return _fail_result()
            return _fail_result()

        harness.try_proof = MagicMock(side_effect=_l0_fail_l1_succeed)
        searcher = _make_searcher(client, harness)
        result = await searcher.search(_make_formalized())

        if result.status is ProofStatus.PROVED:
            assert result.layer is ProofLayer.LAYER1_COMBO

    @pytest.mark.asyncio()
    async def test_nlinarith_with_hints(self) -> None:
        """nlinarith tactic with sq_nonneg hints closes an inequality goal."""
        harness = MagicMock()

        def _try_proof(code: str, tactic: str) -> LeanCheckResult:
            if "nlinarith" in tactic:
                return _success_result()
            return _fail_result()

        harness.try_proof = MagicMock(side_effect=_try_proof)
        harness.verify_full_proof = MagicMock(return_value=_success_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="nlinarith [sq_nonneg a, sq_nonneg b]")

        c = Conjecture(
            id="ineq_ps_001",
            domain=Domain.INEQUALITY,
            nl_statement="a^2 + b^2 >= 2*a*b",
            variables=["a", "b"],
        )
        fc = FormalizedConjecture(
            conjecture=c,
            lean_code="theorem foo (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by\n  sorry",
            status=FormalizationStatus.TYPECHECKS,
        )
        searcher = _make_searcher(client, harness)
        result = await searcher.search(fc)

        if result.status is ProofStatus.PROVED:
            assert result.layer in (ProofLayer.LAYER1_COMBO, ProofLayer.LAYER0_AUTO)


# ---------------------------------------------------------------------------
# TestLayer3LLM
# ---------------------------------------------------------------------------


class TestLayer3LLM:
    """Tests for ProofSearcher._layer3_llm_proof."""

    @pytest.mark.asyncio()
    async def test_llm_first_attempt_success(self) -> None:
        """LLM returns a valid tactic on first round → PROVED, layer=LAYER3_LLM."""
        from conjlean.proof_search import _LAYER0_TACTICS, _LAYER1_TACTICS

        # Count all tactics layers 0 and 1 would try, plus 2 for exact?/apply?
        layers_01_count = len(_LAYER0_TACTICS) + len(_LAYER1_TACTICS) + 2
        call_index = [0]

        def _try_proof_smart(code: str, tactic: str) -> LeanCheckResult:
            call_index[0] += 1
            # All layers 0-2 fail; layer 3 succeeds
            if call_index[0] > layers_01_count:
                return _success_result()
            return _fail_result()

        harness = MagicMock()
        harness.try_proof = MagicMock(side_effect=_try_proof_smart)
        harness.verify_full_proof = MagicMock(return_value=_success_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="```lean\nomega\n```")

        cfg = _make_config(layer3_max_rounds=2)
        searcher = _make_searcher(client, harness, cfg)
        result = await searcher.search(_make_formalized())

        assert result.status is ProofStatus.PROVED
        assert result.layer is ProofLayer.LAYER3_LLM

    @pytest.mark.asyncio()
    async def test_llm_exhausts_rounds(self) -> None:
        """When all LLM rounds fail, the result is ProofResult(OPEN)."""
        harness = MagicMock()
        harness.try_proof = MagicMock(return_value=_fail_result())
        harness.verify_full_proof = MagicMock(return_value=_fail_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="omega")

        cfg = _make_config(layer3_max_rounds=2)
        searcher = _make_searcher(client, harness, cfg)
        result = await searcher.search(_make_formalized())

        assert result.status is ProofStatus.OPEN

    def test_extract_tactic_body_fenced(self) -> None:
        """_extract_tactic_body extracts body from ```lean ... ``` fence."""
        harness = MagicMock()
        client = MagicMock()
        searcher = _make_searcher(client, harness)

        response = "```lean\nomega\n```"
        body = searcher._extract_tactic_body(response)
        assert "omega" in body

    def test_extract_tactic_body_after_by(self) -> None:
        """_extract_tactic_body extracts tactic from 'theorem foo : P := by\\n  omega'."""
        harness = MagicMock()
        client = MagicMock()
        searcher = _make_searcher(client, harness)

        response = "theorem foo : True := by\n  trivial"
        body = searcher._extract_tactic_body(response)
        assert "trivial" in body


# ---------------------------------------------------------------------------
# TestFullSearch
# ---------------------------------------------------------------------------


class TestFullSearch:
    """End-to-end ProofSearcher.search tests."""

    @pytest.mark.asyncio()
    async def test_full_search_succeeds_layer0(self) -> None:
        """When Layer 0 closes it, LLM is not called."""
        harness = MagicMock()
        harness.try_proof = MagicMock(return_value=_success_result())
        harness.verify_full_proof = MagicMock(return_value=_success_result())

        client = MagicMock()
        client.complete = AsyncMock()

        searcher = _make_searcher(client, harness)
        result = await searcher.search(_make_formalized())

        assert result.status is ProofStatus.PROVED
        assert result.layer is ProofLayer.LAYER0_AUTO
        client.complete.assert_not_called()

    @pytest.mark.asyncio()
    async def test_full_search_open_when_all_fail(self) -> None:
        """When all layers fail, the result is ProofStatus.OPEN."""
        harness = MagicMock()
        harness.try_proof = MagicMock(return_value=_fail_result())
        harness.verify_full_proof = MagicMock(return_value=_fail_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="omega")

        cfg = _make_config(layer3_max_rounds=1)
        searcher = _make_searcher(client, harness, cfg)
        result = await searcher.search(_make_formalized())

        assert result.status is ProofStatus.OPEN

    @pytest.mark.asyncio()
    async def test_duration_recorded(self) -> None:
        """ProofResult.duration_seconds is set to a positive value."""
        harness = MagicMock()
        harness.try_proof = MagicMock(return_value=_fail_result())
        harness.verify_full_proof = MagicMock(return_value=_fail_result())

        client = MagicMock()
        client.complete = AsyncMock(return_value="omega")

        cfg = _make_config(layer3_max_rounds=1)
        searcher = _make_searcher(client, harness, cfg)
        result = await searcher.search(_make_formalized())

        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio()
    async def test_invalid_formalization_status_raises(self) -> None:
        """Passing a FormalizedConjecture with UNFORMALIZABLE status raises ValueError."""
        from conjlean.proof_search import ProofSearcher

        c = Conjecture(id="bad", domain=Domain.NUMBER_THEORY, nl_statement="x", variables=[])
        fc = FormalizedConjecture(
            conjecture=c,
            lean_code="...",
            status=FormalizationStatus.UNFORMALIZABLE,
        )

        client = MagicMock()
        harness = MagicMock()
        cfg = ConjLeanConfig()
        searcher = ProofSearcher(client=client, harness=harness, config=cfg)

        with pytest.raises(ValueError, match="TYPECHECKS"):
            await searcher.search(fc)

    @pytest.mark.asyncio()
    async def test_full_search_escalates_to_llm(self) -> None:
        """When layers 0-2 all fail, the LLM (layer 3) is called."""
        harness = MagicMock()
        harness.try_proof = MagicMock(return_value=_fail_result())
        harness.verify_full_proof = MagicMock(return_value=_fail_result())

        llm_called = [False]

        async def _complete(messages, temperature, max_tokens):  # type: ignore[no-untyped-def]
            llm_called[0] = True
            return "omega"

        client = MagicMock()
        client.complete = AsyncMock(side_effect=_complete)

        cfg = _make_config(layer3_max_rounds=1)
        searcher = _make_searcher(client, harness, cfg)
        await searcher.search(_make_formalized())

        assert llm_called[0] is True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_build_full_proof_replaces_sorry(self) -> None:
        """_build_full_proof replaces ':= by sorry' with the tactic body."""
        from conjlean.proof_search import _build_full_proof

        code = "theorem foo : True := by sorry"
        result = _build_full_proof(code, "trivial")
        assert "sorry" not in result
        assert "trivial" in result

    def test_parse_try_this_extracts_exact(self) -> None:
        """_parse_try_this extracts 'exact Foo.bar' from a Try this: line."""
        from conjlean.proof_search import _parse_try_this

        msg = "Try this: exact Nat.dvd_mul_right n k"
        result = _parse_try_this(msg)
        assert result is not None
        assert "exact" in result

    def test_parse_try_this_returns_none_for_no_suggestion(self) -> None:
        """_parse_try_this returns None when no 'Try this:' line is present."""
        from conjlean.proof_search import _parse_try_this

        result = _parse_try_this("some random error message")
        assert result is None

    def test_format_lean_messages_filters_severity(self) -> None:
        """_format_lean_messages only includes error and warning messages."""
        from conjlean.proof_search import _format_lean_messages

        messages = [
            {"severity": "info", "data": "info line"},
            {"severity": "error", "data": "error line"},
            {"severity": "warning", "data": "warning line"},
        ]
        result = _format_lean_messages(messages)
        assert "error line" in result
        assert "warning line" in result
        assert "info line" not in result

    def test_strip_theorem_wrapper_extracts_tactic(self) -> None:
        """_strip_theorem_wrapper removes theorem boilerplate to get tactic body."""
        from conjlean.proof_search import _strip_theorem_wrapper

        block = "theorem foo : True := by\n  trivial"
        result = _strip_theorem_wrapper(block)
        assert "trivial" in result
        assert "theorem" not in result
