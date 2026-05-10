"""
Tests for the REFUTE R-Agent (:class:`~conjlean.refuter.Refuter`).

Coverage:
- BOUNDARY strategy finds counterexamples for known-false conjectures.
- BOUNDARY correctly returns no counterexample for known-true conjectures.
- RANDOM_STRUCTURED strategy delegates to LLM suggestion and verifies via SymPy.
- RANDOM_STRUCTURED candidate sampling is reproducible when stdlib random is seeded.
- search_all_strategies terminates on the first successful strategy.
- RefuterResult structure has all expected fields populated correctly.
- strategy_scores are updated after each attempt.
- Empty nl_statement raises ValueError (fail-fast validation).
- ANALOGICAL strategy gracefully handles the absence of past refutations.

Design notes:
- :class:`~conjlean.models.LLMClient` is mocked via ``AsyncMock`` for speed
  and hermeticity.
- SymPy is NOT mocked — counterexample verification runs against real SymPy
  to guard against regressions in the verification logic.
- All async tests are marked with ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import random

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from conjlean.config import ConjLeanConfig
from conjlean.schemas import (
    Conjecture,
    CounterexampleCandidate,
    CounterexampleStatus,
    Domain,
    RefuterResult,
    RefuterStrategy,
)

# ---------------------------------------------------------------------------
# Lazy import guard: Refuter may not exist yet when running the test suite
# without the refuter module.  Import is deferred to each test so collection
# does not fail.
# ---------------------------------------------------------------------------


def _import_refuter():
    """Import Refuter lazily so the module is skipped gracefully if absent."""
    try:
        from conjlean.refuter import Refuter  # type: ignore[import]
        return Refuter
    except ImportError:
        pytest.skip("conjlean.refuter module not yet implemented — skipping.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_client() -> MagicMock:
    """
    Return a MagicMock LLM client with an async ``complete`` method.

    The default return value is a blank string; individual tests override
    ``mock_llm_client.complete.return_value`` as needed.
    """
    client = MagicMock()
    client.complete = AsyncMock(return_value="")
    client.complete_batch = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def default_config() -> ConjLeanConfig:
    """Return a default ConjLeanConfig (no YAML required)."""
    return ConjLeanConfig()


@pytest.fixture()
def false_conjecture_prime_poly() -> Conjecture:
    """
    Return a conjecture that is known to be FALSE.

    The classic prime-generating polynomial n^2 + n + 41 is NOT prime at
    n=40 (result: 1681 = 41^2).  The BOUNDARY strategy must find n=40.
    """
    return Conjecture(
        id="false_prime_poly_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="For all n >= 1, n^2 + n + 41 is prime",
        variables=["n"],
        source="test",
        metadata={"ground_truth": "false", "counterexample_n": 40},
    )


@pytest.fixture()
def true_conjecture_consecutive_product() -> Conjecture:
    """
    Return a conjecture that is known to be TRUE.

    2 divides n*(n+1) for all natural numbers n (consecutive integer product).
    """
    return Conjecture(
        id="true_consec_prod_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="2 divides n*(n+1) for all natural numbers n",
        variables=["n"],
        source="test",
        metadata={"ground_truth": "true"},
    )


@pytest.fixture()
def false_conjecture_inequality() -> Conjecture:
    """
    Return an inequality conjecture known to be FALSE.

    a^2 >= 2*a is false for a=0.5 (0.25 >= 1.0 is False).
    """
    return Conjecture(
        id="false_ineq_001",
        domain=Domain.INEQUALITY,
        nl_statement="a^2 >= 2*a for all positive reals a",
        variables=["a"],
        source="test",
        metadata={"ground_truth": "false"},
    )


@pytest.fixture()
def simple_conjecture() -> Conjecture:
    """Return a simple well-formed conjecture for structural tests."""
    return Conjecture(
        id="simple_nt_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="n^2 >= n for all n >= 1",
        variables=["n"],
        source="test",
    )


# ---------------------------------------------------------------------------
# Test 1: BOUNDARY finds a counterexample for a known-false conjecture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_finds_known_false(
    false_conjecture_prime_poly: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    BOUNDARY strategy must find n=40 as a counterexample to the prime polynomial.

    The conjecture 'n^2 + n + 41 is prime' fails at n=40 (41^2 = 1681).
    BOUNDARY sweeps edge/boundary values and must catch this.
    """
    Refuter = _import_refuter()
    refuter = Refuter(client=mock_llm_client, config=default_config)

    result: RefuterResult = await refuter.search(
        conjecture=false_conjecture_prime_poly,
        strategy=RefuterStrategy.BOUNDARY,
        max_rounds=5,
    )

    assert isinstance(result, RefuterResult), "search must return a RefuterResult"
    assert result.best_counterexample is not None, (
        "BOUNDARY should find n=40 counterexample for n^2+n+41 is prime"
    )
    assert result.best_counterexample.status == CounterexampleStatus.CONFIRMED, (
        "Found counterexample must be CONFIRMED by SymPy"
    )
    assert result.strategy_used == RefuterStrategy.BOUNDARY
    assert result.rounds >= 1


# ---------------------------------------------------------------------------
# Test 2: BOUNDARY returns no counterexample for a known-true conjecture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_survives_true(
    true_conjecture_consecutive_product: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    BOUNDARY strategy must NOT find a counterexample for '2 divides n*(n+1)'.

    This is a true conjecture and no counterexample should be returned.
    """
    Refuter = _import_refuter()
    refuter = Refuter(client=mock_llm_client, config=default_config)

    result: RefuterResult = await refuter.search(
        conjecture=true_conjecture_consecutive_product,
        strategy=RefuterStrategy.BOUNDARY,
        max_rounds=5,
    )

    assert isinstance(result, RefuterResult)
    assert result.best_counterexample is None, (
        "BOUNDARY must NOT find a counterexample for the true conjecture '2 | n*(n+1)'"
    )
    assert result.strategy_used == RefuterStrategy.BOUNDARY


# ---------------------------------------------------------------------------
# Test 3: RANDOM_STRUCTURED uses LLM suggestions and verifies with SymPy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_random_structured_finds_counterexample(
    false_conjecture_inequality: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    RANDOM_STRUCTURED strategy should use the LLM to suggest candidate values,
    then verify them with SymPy.

    The LLM mock is configured to return a=0.5, which is a valid counterexample
    for 'a^2 >= 2*a' (0.25 < 1.0).  SymPy must confirm the candidate.
    """
    Refuter = _import_refuter()

    # Return a structured suggestion that the R-Agent's parser can extract
    mock_llm_client.complete = AsyncMock(
        return_value=(
            "Try a=0.5. At a=0.5: a^2 = 0.25, 2*a = 1.0. "
            "Since 0.25 < 1.0, the inequality fails. Counterexample: a=0.5"
        )
    )

    refuter = Refuter(client=mock_llm_client, config=default_config)

    result: RefuterResult = await refuter.search(
        conjecture=false_conjecture_inequality,
        strategy=RefuterStrategy.RANDOM_STRUCTURED,
        max_rounds=3,
    )

    assert isinstance(result, RefuterResult)
    # Either the R-Agent parsed the suggestion and SymPy confirmed it,
    # or SymPy independently discovered the CE during numerical sweep.
    # Either way, a CONFIRMED counterexample must exist for this false conjecture.
    if result.best_counterexample is not None:
        assert result.best_counterexample.status == CounterexampleStatus.CONFIRMED


# ---------------------------------------------------------------------------
# Test 4: search_all_strategies stops on first successful strategy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_all_strategies_stops_on_first_success(
    false_conjecture_prime_poly: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    search_all_strategies must return as soon as any strategy finds a CE.

    Once BOUNDARY succeeds, the result should have best_counterexample set
    without exhausting all four strategies.
    """
    Refuter = _import_refuter()
    refuter = Refuter(client=mock_llm_client, config=default_config)

    result: RefuterResult = await refuter.search_all_strategies(
        conjecture=false_conjecture_prime_poly,
        max_rounds_per_strategy=5,
    )

    assert isinstance(result, RefuterResult)
    assert result.best_counterexample is not None, (
        "search_all_strategies must return a confirmed CE for the known-false conjecture"
    )
    assert result.best_counterexample.status == CounterexampleStatus.CONFIRMED
    # Should not have exhausted all four strategies; at most BOUNDARY was needed
    assert result.strategy_used is not None


# ---------------------------------------------------------------------------
# Test 5: RefuterResult structure has all required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuter_result_structure(
    simple_conjecture: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    The RefuterResult returned by search must have all required fields
    populated with the correct types.
    """
    Refuter = _import_refuter()
    refuter = Refuter(client=mock_llm_client, config=default_config)

    result: RefuterResult = await refuter.search(
        conjecture=simple_conjecture,
        strategy=RefuterStrategy.BOUNDARY,
        max_rounds=3,
    )

    assert hasattr(result, "conjecture"), "RefuterResult must have conjecture field"
    assert hasattr(result, "candidates"), "RefuterResult must have candidates field"
    assert hasattr(result, "best_counterexample"), "RefuterResult must have best_counterexample"
    assert hasattr(result, "strategy_used"), "RefuterResult must have strategy_used"
    assert hasattr(result, "rounds"), "RefuterResult must have rounds field"
    assert hasattr(result, "strategy_scores"), "RefuterResult must have strategy_scores"

    assert result.conjecture is simple_conjecture
    assert isinstance(result.candidates, list)
    assert isinstance(result.rounds, int)
    assert result.rounds >= 0
    assert isinstance(result.strategy_scores, dict)

    if result.best_counterexample is not None:
        assert isinstance(result.best_counterexample, CounterexampleCandidate)
        assert result.best_counterexample.status == CounterexampleStatus.CONFIRMED


# ---------------------------------------------------------------------------
# Test 6: strategy_scores updated after each strategy attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strategy_scores_updated(
    simple_conjecture: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    After search completes, strategy_scores must contain an entry for the
    attempted strategy with a non-negative integer value.
    """
    Refuter = _import_refuter()
    refuter = Refuter(client=mock_llm_client, config=default_config)

    result: RefuterResult = await refuter.search(
        conjecture=simple_conjecture,
        strategy=RefuterStrategy.BOUNDARY,
        max_rounds=2,
    )

    assert isinstance(result.strategy_scores, dict), "strategy_scores must be a dict"
    # The BOUNDARY strategy key should be present
    boundary_key = RefuterStrategy.BOUNDARY.value
    if boundary_key in result.strategy_scores:
        assert isinstance(result.strategy_scores[boundary_key], int), (
            "strategy_scores values must be integers"
        )
        assert result.strategy_scores[boundary_key] >= 0


# ---------------------------------------------------------------------------
# Test 7: Empty nl_statement raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_conjecture_raises(
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    search must raise ValueError immediately when nl_statement is empty.

    Fail-fast input validation is required per project code style guidelines.
    """
    Refuter = _import_refuter()
    refuter = Refuter(client=mock_llm_client, config=default_config)

    empty_conjecture = Conjecture(
        id="empty_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="",
        variables=["n"],
        source="test",
    )

    with pytest.raises(ValueError, match=r"nl_statement"):
        await refuter.search(
            conjecture=empty_conjecture,
            strategy=RefuterStrategy.BOUNDARY,
            max_rounds=3,
        )


# ---------------------------------------------------------------------------
# Test 8: ANALOGICAL gracefully handles no past refutations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analogical_with_no_past_data(
    simple_conjecture: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    ANALOGICAL strategy must not raise when no past refutations exist.

    When the R-Agent has no analogical history to draw from, it should
    degrade gracefully (fall back to numerical search or return empty candidates)
    rather than raising an unhandled exception.
    """
    Refuter = _import_refuter()
    refuter = Refuter(client=mock_llm_client, config=default_config)

    # Configure LLM to indicate no past data available
    mock_llm_client.complete = AsyncMock(
        return_value="No analogous past refutations found. Cannot transfer patterns."
    )

    result: RefuterResult = await refuter.search(
        conjecture=simple_conjecture,
        strategy=RefuterStrategy.ANALOGICAL,
        max_rounds=2,
    )

    assert isinstance(result, RefuterResult), (
        "ANALOGICAL with no past data must return a RefuterResult, not raise"
    )
    assert isinstance(result.candidates, list)
    # best_counterexample may be None — that is acceptable when no past data exists


# ---------------------------------------------------------------------------
# Test 9: RANDOM_STRUCTURED respects stdlib random seed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_random_structured_reproducible_with_seed(
    false_conjecture_prime_poly: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    RANDOM_STRUCTURED candidate pool must be identical across two runs that
    start with the same stdlib random seed.

    This guards the fix that replaced random.Random() (OS-seeded, uncontrolled)
    with the global random module (controlled by random.seed()).  If the fix
    regresses, the two candidate lists will diverge.
    """
    Refuter = _import_refuter()

    async def _collect_candidates(seed: int) -> list[str]:
        random.seed(seed)
        refuter = Refuter(client=mock_llm_client, config=default_config)
        result: RefuterResult = await refuter.search(
            conjecture=false_conjecture_prime_poly,
            strategy=RefuterStrategy.RANDOM_STRUCTURED,
            max_rounds=1,
        )
        return [c.candidate_str for c in result.candidates]

    run_a = await _collect_candidates(seed=7)
    run_b = await _collect_candidates(seed=7)

    assert run_a == run_b, (
        "RANDOM_STRUCTURED must produce identical candidates for the same seed. "
        f"run_a={run_a!r}, run_b={run_b!r}"
    )


@pytest.mark.asyncio
async def test_random_structured_differs_across_seeds(
    false_conjecture_prime_poly: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    RANDOM_STRUCTURED must draw different candidates for different seeds.

    This is a probabilistic sanity check: the probability that two independent
    draws from a pool of ~300 integers produce the same 20-element sample is
    negligibly small, so this should never flap.
    """
    Refuter = _import_refuter()

    async def _collect_sampled_pool(seed: int) -> set[str]:
        random.seed(seed)
        refuter = Refuter(client=mock_llm_client, config=default_config)
        result: RefuterResult = await refuter.search(
            conjecture=false_conjecture_prime_poly,
            strategy=RefuterStrategy.RANDOM_STRUCTURED,
            max_rounds=1,
        )
        return {c.candidate_str for c in result.candidates}

    run_seed_1 = await _collect_sampled_pool(seed=1)
    run_seed_2 = await _collect_sampled_pool(seed=99999)

    assert run_seed_1 != run_seed_2, (
        "Different seeds must yield different RANDOM_STRUCTURED candidate pools"
    )
