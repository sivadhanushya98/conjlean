"""
Tests for the REFUTE loop (:class:`~conjlean.refute_loop.RefuteLoop`).

Coverage:
1. Loop correctly terminates with REFUTED when R-Agent returns a counterexample.
2. Loop invokes C-Agent refinement when a counterexample is found and max_refinements > 0.
3. Loop correctly terminates with SURVIVED when R-Agent never finds a counterexample.
4. Loop terminates with BUDGET_EXHAUSTED when S-Agent signals stop at max_rounds=2.
5. run_batch with 3 conjectures returns 3 RefuteLoopResult objects.
6. run_batch respects max_concurrent via asyncio.Semaphore (concurrency counter proof).
7. max_refinements cap is respected — only 1 refinement even if multiple CEs are found.
8. RefuteLoopResult.refuter_results is populated with one entry per R-Agent round.

Design notes:
- Refuter, Strategist, and LLM client are fully mocked with ``AsyncMock``/``MagicMock``.
- The Refuter mock exposes ``run(conjecture, strategy) -> RefuterResult`` matching the
  actual interface called by ``RefuteLoop._run_refuter_safely``.
- All async tests are decorated with ``@pytest.mark.asyncio``.
- Each test is designed to complete in well under 5 seconds.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from conjlean.config import ConjLeanConfig
from conjlean.schemas import (
    Conjecture,
    ConjectureRefinement,
    CounterexampleCandidate,
    CounterexampleStatus,
    Domain,
    RefuteLoopResult,
    RefuteLoopStatus,
    RefuterResult,
    RefuterStrategy,
)

# ---------------------------------------------------------------------------
# Lazy import guard — skip cleanly if module is not yet on the import path
# ---------------------------------------------------------------------------


def _import_refute_loop():
    """Import RefuteLoop lazily; skip the whole test if the module is absent."""
    try:
        from conjlean.refute_loop import RefuteLoop  # type: ignore[import]

        return RefuteLoop
    except ImportError:
        pytest.skip("conjlean.refute_loop not importable — skipping.")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> ConjLeanConfig:
    """Return a default ConjLeanConfig (no YAML file required)."""
    return ConjLeanConfig()


@pytest.fixture()
def mock_llm_client() -> MagicMock:
    """
    Return a MagicMock LLM client whose ``complete`` coroutine returns a
    well-formed JSON refinement reply so that ``_parse_refinement_reply`` succeeds.
    """
    client = MagicMock()
    client.complete = AsyncMock(
        return_value=(
            '{"refined_statement": "For all n >= 1 with n != 40, '
            'n^2 + n + 41 is prime", '
            '"refinement_type": "added_condition", '
            '"explanation": "Excludes the counterexample at n=40."}'
        )
    )
    client.complete_batch = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def conjecture_false() -> Conjecture:
    """A conjecture that is known to be false (n^2 + n + 41 is prime ∀ n ≥ 1)."""
    return Conjecture(
        id="loop_test_false_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="For all n >= 1, n^2 + n + 41 is prime",
        variables=["n"],
        source="test",
    )


@pytest.fixture()
def conjecture_true() -> Conjecture:
    """A conjecture that is known to be true (2 divides n*(n+1))."""
    return Conjecture(
        id="loop_test_true_001",
        domain=Domain.NUMBER_THEORY,
        nl_statement="2 divides n*(n+1) for all natural numbers n",
        variables=["n"],
        source="test",
    )


@pytest.fixture()
def confirmed_candidate(conjecture_false: Conjecture) -> CounterexampleCandidate:
    """A CounterexampleCandidate with status CONFIRMED."""
    return CounterexampleCandidate(
        conjecture_id=conjecture_false.id,
        candidate_str="n=40: 40^2 + 40 + 41 = 1681 = 41^2, not prime",
        strategy=RefuterStrategy.BOUNDARY,
        status=CounterexampleStatus.CONFIRMED,
        evidence={"n=40": False},
        reasoning="Boundary sweep at n=40 yields composite 1681.",
    )


def _make_ce_result(
    conjecture: Conjecture,
    candidate: CounterexampleCandidate,
) -> RefuterResult:
    """Build a RefuterResult containing the supplied confirmed counterexample."""
    return RefuterResult(
        conjecture=conjecture,
        candidates=[candidate],
        best_counterexample=candidate,
        strategy_used=RefuterStrategy.BOUNDARY,
        rounds=1,
        strategy_scores={RefuterStrategy.BOUNDARY.value: 1},
    )


def _make_no_ce_result(conjecture: Conjecture) -> RefuterResult:
    """Build a RefuterResult with no counterexample found."""
    return RefuterResult(
        conjecture=conjecture,
        candidates=[],
        best_counterexample=None,
        strategy_used=RefuterStrategy.BOUNDARY,
        rounds=1,
        strategy_scores={RefuterStrategy.BOUNDARY.value: 0},
    )


def _make_strategist(
    *,
    should_stop_return: tuple[bool, str] = (False, ""),
    should_stop_side_effect=None,
) -> MagicMock:
    """
    Build a minimally functional Strategist mock.

    Args:
        should_stop_return: Static return value for ``should_stop``.
        should_stop_side_effect: Optional side_effect sequence / callable for
            ``should_stop`` (overrides ``should_stop_return``).

    Returns:
        A MagicMock with all methods called by RefuteLoop wired up.
    """
    strategist = MagicMock()
    strategist.select_strategy = AsyncMock(return_value=RefuterStrategy.BOUNDARY)
    strategist.get_stats_summary = MagicMock(return_value={})
    strategist.update_stats = MagicMock()
    if should_stop_side_effect is not None:
        strategist.should_stop = MagicMock(side_effect=should_stop_side_effect)
    else:
        strategist.should_stop = MagicMock(return_value=should_stop_return)
    return strategist


# ---------------------------------------------------------------------------
# Test 1 — Loop correctly terminates REFUTED when R-Agent returns a CE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_refutes_false_conjecture(
    conjecture_false: Conjecture,
    confirmed_candidate: CounterexampleCandidate,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    When the R-Agent returns a confirmed counterexample and max_refinements=0,
    the loop must terminate with status REFUTED and store the counterexample.
    """
    RefuteLoop = _import_refute_loop()

    refuter = MagicMock()
    refuter.run = AsyncMock(
        return_value=_make_ce_result(conjecture_false, confirmed_candidate)
    )

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=_make_strategist(),
        config=default_config,
    )

    result: RefuteLoopResult = await loop.run_single(
        conjecture=conjecture_false,
        max_rounds=10,
        max_refinements=0,  # no refinement → pure REFUTED path
    )

    assert isinstance(result, RefuteLoopResult)
    assert result.status == RefuteLoopStatus.REFUTED, (
        f"Expected REFUTED, got {result.status}"
    )
    assert result.confirmed_counterexample is not None, (
        "confirmed_counterexample must be populated for a REFUTED result"
    )
    assert result.confirmed_counterexample.status == CounterexampleStatus.CONFIRMED


# ---------------------------------------------------------------------------
# Test 2 — Loop refines the conjecture after finding a counterexample
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_refines_conjecture(
    conjecture_false: Conjecture,
    confirmed_candidate: CounterexampleCandidate,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    When the R-Agent finds a counterexample and max_refinements >= 1, the loop
    must invoke the C-Agent (LLM), append a ConjectureRefinement, and produce a
    final_conjecture whose nl_statement differs from the original.
    """
    RefuteLoop = _import_refute_loop()

    # Round 1: CE found → triggers refinement.
    # Round 2+: no CE found so loop reaches SURVIVED after refinement.
    call_counter: dict[str, int] = {"n": 0}

    async def _refuter_run(conjecture: Conjecture, strategy: RefuterStrategy) -> RefuterResult:
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return _make_ce_result(conjecture, confirmed_candidate)
        return _make_no_ce_result(conjecture)

    refuter = MagicMock()
    refuter.run = AsyncMock(side_effect=_refuter_run)

    strategist = _make_strategist(should_stop_return=(True, "strategies_exhausted"))

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=strategist,
        config=default_config,
    )

    result: RefuteLoopResult = await loop.run_single(
        conjecture=conjecture_false,
        max_rounds=10,
        max_refinements=1,
    )

    assert isinstance(result, RefuteLoopResult)
    assert len(result.refinements) == 1, (
        f"Expected exactly 1 refinement, got {len(result.refinements)}"
    )
    refinement = result.refinements[0]
    assert isinstance(refinement, ConjectureRefinement)
    assert refinement.refined_statement, "refined_statement must be non-empty"

    assert result.final_conjecture is not None
    assert result.final_conjecture.nl_statement != conjecture_false.nl_statement, (
        "final_conjecture must differ from original after C-Agent refinement"
    )


# ---------------------------------------------------------------------------
# Test 3 — Loop survives a true conjecture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_survives_true_conjecture(
    conjecture_true: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    When the R-Agent never finds a counterexample and the S-Agent signals stop
    after the first round, the loop must terminate with SURVIVED and set
    confirmed_counterexample to None.
    """
    RefuteLoop = _import_refute_loop()

    refuter = MagicMock()
    refuter.run = AsyncMock(return_value=_make_no_ce_result(conjecture_true))

    # Strategist signals stop on the first should_stop check
    strategist = _make_strategist(should_stop_return=(True, "all_strategies_exhausted"))

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=strategist,
        config=default_config,
    )

    result: RefuteLoopResult = await loop.run_single(
        conjecture=conjecture_true,
        max_rounds=10,
        max_refinements=3,
    )

    assert isinstance(result, RefuteLoopResult)
    assert result.status == RefuteLoopStatus.SURVIVED, (
        f"Expected SURVIVED for true conjecture, got {result.status}"
    )
    assert result.confirmed_counterexample is None, (
        "confirmed_counterexample must be None for a SURVIVED result"
    )


# ---------------------------------------------------------------------------
# Test 4 — Loop terminates BUDGET_EXHAUSTED when S-Agent always says stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_budget_exhausted(
    conjecture_false: Conjecture,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    With max_rounds=2, a Refuter that never finds a CE, and a Strategist that
    always signals stop with total_rounds >= max_rounds reached, the loop must
    terminate with BUDGET_EXHAUSTED and total_rounds <= 2.
    """
    RefuteLoop = _import_refute_loop()

    refuter = MagicMock()
    refuter.run = AsyncMock(return_value=_make_no_ce_result(conjecture_false))

    # should_stop always returns True AND total_rounds will equal max_rounds=2,
    # so the loop enters the BUDGET_EXHAUSTED branch in run_single.
    strategist = _make_strategist(
        should_stop_side_effect=[
            (True, "budget_exhausted_max_rounds=2"),
            (True, "budget_exhausted_max_rounds=2"),
            (True, "budget_exhausted_max_rounds=2"),
        ]
    )

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=strategist,
        config=default_config,
    )

    result: RefuteLoopResult = await loop.run_single(
        conjecture=conjecture_false,
        max_rounds=2,
        max_refinements=3,
    )

    assert isinstance(result, RefuteLoopResult)
    assert result.status == RefuteLoopStatus.BUDGET_EXHAUSTED, (
        f"Expected BUDGET_EXHAUSTED, got {result.status}"
    )
    assert result.total_rounds <= 2, (
        f"total_rounds must be <= max_rounds=2, got {result.total_rounds}"
    )


# ---------------------------------------------------------------------------
# Test 5 — run_batch with 3 conjectures returns 3 results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_returns_all_results(
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    run_batch on 3 conjectures must return a list of exactly 3 RefuteLoopResult
    objects, one per input conjecture.
    """
    RefuteLoop = _import_refute_loop()

    conjectures = [
        Conjecture(
            id=f"batch_conj_{i:03d}",
            domain=Domain.NUMBER_THEORY,
            nl_statement=f"For all n, n + {i} > n",
            variables=["n"],
            source="test",
        )
        for i in range(3)
    ]

    async def _no_ce_run(conjecture: Conjecture, strategy: RefuterStrategy) -> RefuterResult:
        return _make_no_ce_result(conjecture)

    refuter = MagicMock()
    refuter.run = AsyncMock(side_effect=_no_ce_run)

    strategist = _make_strategist(should_stop_return=(True, "exhausted"))

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=strategist,
        config=default_config,
    )

    results: list[RefuteLoopResult] = await loop.run_batch(
        conjectures=conjectures,
        max_rounds=3,
        max_refinements=0,
        max_concurrent=3,
    )

    assert isinstance(results, list), "run_batch must return a list"
    assert len(results) == 3, f"Expected 3 results for 3 conjectures, got {len(results)}"
    for result in results:
        assert isinstance(result, RefuteLoopResult), (
            "Every element of run_batch output must be a RefuteLoopResult"
        )


# ---------------------------------------------------------------------------
# Test 6 — run_batch respects max_concurrent via Semaphore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_respects_max_concurrent(
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    run_batch with max_concurrent=2 must never have more than 2 run_single
    coroutines executing simultaneously.  A shared concurrent-count counter
    is incremented at entry and decremented at exit of each simulated R-Agent
    call; the peak value must not exceed max_concurrent=2.
    """
    RefuteLoop = _import_refute_loop()

    max_concurrent = 2
    n_conjectures = 6

    conjectures = [
        Conjecture(
            id=f"concur_conj_{i:03d}",
            domain=Domain.COMBINATORICS,
            nl_statement=f"C(2*n, n) >= {i} for all n >= 1",
            variables=["n"],
            source="test",
        )
        for i in range(n_conjectures)
    ]

    peak_concurrent: dict[str, int] = {"current": 0, "peak": 0}

    async def _tracked_run(conjecture: Conjecture, strategy: RefuterStrategy) -> RefuterResult:
        peak_concurrent["current"] += 1
        if peak_concurrent["current"] > peak_concurrent["peak"]:
            peak_concurrent["peak"] = peak_concurrent["current"]
        # Yield control so other coroutines can enter — ensures interleaving.
        await asyncio.sleep(0)
        peak_concurrent["current"] -= 1
        return _make_no_ce_result(conjecture)

    refuter = MagicMock()
    refuter.run = AsyncMock(side_effect=_tracked_run)

    strategist = _make_strategist(should_stop_return=(True, "exhausted"))

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=strategist,
        config=default_config,
    )

    results: list[RefuteLoopResult] = await loop.run_batch(
        conjectures=conjectures,
        max_rounds=1,
        max_refinements=0,
        max_concurrent=max_concurrent,
    )

    assert len(results) == n_conjectures, (
        f"All {n_conjectures} conjectures must complete, got {len(results)}"
    )
    assert peak_concurrent["peak"] <= max_concurrent, (
        f"Peak concurrent R-Agent calls {peak_concurrent['peak']} exceeded "
        f"max_concurrent={max_concurrent} — Semaphore not effective"
    )


# ---------------------------------------------------------------------------
# Test 7 — max_refinements cap: only 1 refinement even with multiple CEs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refinement_count_cap(
    conjecture_false: Conjecture,
    confirmed_candidate: CounterexampleCandidate,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    With max_refinements=1, the loop must apply at most 1 C-Agent refinement
    even if the R-Agent keeps returning confirmed counterexamples across
    multiple rounds.
    """
    RefuteLoop = _import_refute_loop()

    # Refuter always returns a confirmed CE, regardless of which conjecture it receives.
    async def _always_ce(conjecture: Conjecture, strategy: RefuterStrategy) -> RefuterResult:
        candidate = CounterexampleCandidate(
            conjecture_id=conjecture.id,
            candidate_str="n=40: composite 1681",
            strategy=strategy,
            status=CounterexampleStatus.CONFIRMED,
            evidence={"n=40": False},
        )
        return RefuterResult(
            conjecture=conjecture,
            candidates=[candidate],
            best_counterexample=candidate,
            strategy_used=strategy,
            rounds=1,
            strategy_scores={strategy.value: 1},
        )

    refuter = MagicMock()
    refuter.run = AsyncMock(side_effect=_always_ce)

    # Strategist never stops early, giving the loop full 10 rounds to try more refinements.
    strategist = _make_strategist(should_stop_return=(False, ""))

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=strategist,
        config=default_config,
    )

    result: RefuteLoopResult = await loop.run_single(
        conjecture=conjecture_false,
        max_rounds=10,
        max_refinements=1,
    )

    assert isinstance(result, RefuteLoopResult)
    assert len(result.refinements) <= 1, (
        f"max_refinements=1 must cap refinements at 1, got {len(result.refinements)}"
    )


# ---------------------------------------------------------------------------
# Test 8 — RefuteLoopResult.refuter_results is populated with correct count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_result_has_trajectory(
    conjecture_false: Conjecture,
    confirmed_candidate: CounterexampleCandidate,
    mock_llm_client: MagicMock,
    default_config: ConjLeanConfig,
) -> None:
    """
    RefuteLoopResult.refuter_results must contain exactly one RefuterResult
    per R-Agent round executed.  In this test the Refuter returns no-CE for
    the first 2 rounds then a confirmed CE on round 3, so refuter_results
    must have length >= 3 and at least one entry must contain a confirmed
    best_counterexample.
    """
    RefuteLoop = _import_refute_loop()

    call_counter: dict[str, int] = {"n": 0}

    async def _sequenced_run(conjecture: Conjecture, strategy: RefuterStrategy) -> RefuterResult:
        call_counter["n"] += 1
        if call_counter["n"] < 3:
            return _make_no_ce_result(conjecture)
        # Round 3: return confirmed CE
        candidate = CounterexampleCandidate(
            conjecture_id=conjecture.id,
            candidate_str="n=40: composite 1681",
            strategy=strategy,
            status=CounterexampleStatus.CONFIRMED,
            evidence={"n=40": False},
        )
        return RefuterResult(
            conjecture=conjecture,
            candidates=[candidate],
            best_counterexample=candidate,
            strategy_used=strategy,
            rounds=1,
            strategy_scores={strategy.value: 1},
        )

    refuter = MagicMock()
    refuter.run = AsyncMock(side_effect=_sequenced_run)

    # Strategist cycles through three strategies across the first 3 rounds.
    strategies = [
        RefuterStrategy.BOUNDARY,
        RefuterStrategy.RANDOM_STRUCTURED,
        RefuterStrategy.SYMBOLIC_PERTURBATION,
        RefuterStrategy.ANALOGICAL,
    ]
    strategy_iter = iter(strategies)

    async def _next_strategy(*args, **kwargs) -> RefuterStrategy:
        try:
            return next(strategy_iter)
        except StopIteration:
            return RefuterStrategy.BOUNDARY

    strategist = MagicMock()
    strategist.select_strategy = AsyncMock(side_effect=_next_strategy)
    strategist.get_stats_summary = MagicMock(return_value={})
    strategist.should_stop = MagicMock(return_value=(False, ""))
    strategist.update_stats = MagicMock()

    loop = RefuteLoop(
        client=mock_llm_client,
        refuter=refuter,
        strategist=strategist,
        config=default_config,
    )

    result: RefuteLoopResult = await loop.run_single(
        conjecture=conjecture_false,
        max_rounds=10,
        max_refinements=0,  # no refinement so loop stops immediately on CE
    )

    assert isinstance(result, RefuteLoopResult)
    assert len(result.refuter_results) >= 3, (
        f"refuter_results must have >= 3 entries (one per round), "
        f"got {len(result.refuter_results)}"
    )
    for rr in result.refuter_results:
        assert isinstance(rr, RefuterResult), (
            "Every element of refuter_results must be a RefuterResult"
        )

    confirmed_rounds = [
        rr for rr in result.refuter_results
        if rr.best_counterexample is not None
        and rr.best_counterexample.status == CounterexampleStatus.CONFIRMED
    ]
    assert len(confirmed_rounds) >= 1, (
        "At least one refuter_results entry must contain the confirmed counterexample"
    )
