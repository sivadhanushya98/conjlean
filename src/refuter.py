"""
R-Agent: Counterexample generation engine for the REFUTE pipeline.

This module implements the core paper contribution of REFUTE (ICML AI4Research
2026) — a multi-strategy search agent that finds counterexamples to mathematical
conjectures using four complementary paradigms:

1. BOUNDARY: Systematic probing of edge cases and domain extremes.
2. RANDOM_STRUCTURED: Mathematically-informed random sampling.
3. ANALOGICAL: LLM-guided transfer from past successful refutations.
4. SYMBOLIC_PERTURBATION: Parameter perturbation to locate statement boundaries.

All numerical verification is performed inline via SymPy in async-compatible
worker threads.  LLM calls are rate-limited to ``max_concurrent`` concurrent
coroutines via an ``asyncio.Semaphore``.  Strategy-level outcomes are tracked
in a ``strategy_scores`` dict for consumption by the S-Agent meta-controller.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import random
import re
from typing import Any, Optional

from tqdm import tqdm

from conjlean.config import ConjLeanConfig
from conjlean.models import LLMClient
from conjlean.schemas import (
    Conjecture,
    CounterexampleCandidate,
    CounterexampleStatus,
    Domain,
    RefuterResult,
    RefuterStrategy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_FIRST_20_PRIMES: list[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
]

_FIBONACCI_NUMBERS: list[int] = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
]

_LLM_TEMPERATURE: float = 0.7
_LLM_MAX_TOKENS: int = 1024
_SYMPY_EVAL_TIMEOUT: float = 4.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_llm_candidates(raw: str) -> list[dict[str, Any]]:
    """Parse candidate counterexamples from an LLM completion string.

    Attempts JSON parsing first; falls back to line-by-line extraction of
    numeric values if JSON parsing fails.  The robust fallback ensures that
    even malformed LLM outputs yield actionable candidates rather than
    silently failing.

    Args:
        raw: Raw completion text from an LLM call.

    Returns:
        A list of dicts, each with at least a ``"value"`` key (str) and
        optionally a ``"reasoning"`` key (str).  May be empty if no
        parseable content is found.
    """
    raw = raw.strip()

    # Attempt 1: full JSON parse.
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "candidates" in data:
            raw_candidates = data["candidates"]
            if isinstance(raw_candidates, list):
                return [
                    c for c in raw_candidates if isinstance(c, dict) and "value" in c
                ]
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: extract a JSON block embedded in surrounding prose.
    json_block_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_block_match:
        try:
            data = json.loads(json_block_match.group(0))
            if isinstance(data, dict) and "candidates" in data:
                raw_candidates = data["candidates"]
                if isinstance(raw_candidates, list):
                    return [
                        c for c in raw_candidates
                        if isinstance(c, dict) and "value" in c
                    ]
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3: line-by-line extraction of numeric tokens.
    candidates: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        # Grab first numeric token (int, float, or tuple-like).
        token_match = re.search(
            r"(?:\([\d\s,.\-]+\)|\-?\d+(?:\.\d+)?(?:e[\+\-]?\d+)?)", line
        )
        if token_match:
            candidates.append({"value": token_match.group(0), "reasoning": line})
    return candidates


def _extract_numeric_value(value_str: str) -> Optional[Any]:
    """Convert a candidate value string to a Python numeric type.

    Handles integers, floats, and simple tuples of numbers.  Returns
    ``None`` when parsing fails so callers can skip gracefully.

    Args:
        value_str: String representation from an LLM candidate dict.

    Returns:
        An ``int``, ``float``, or ``tuple`` of numbers, or ``None``.
    """
    value_str = str(value_str).strip()

    # Tuple / comma-separated list.
    tuple_match = re.fullmatch(r"\(?([\d\s,.\-e]+)\)?", value_str)
    if tuple_match and "," in value_str:
        parts = [p.strip() for p in value_str.strip("()").split(",")]
        try:
            nums = [float(p) for p in parts if p]
            if all(n == int(n) for n in nums):
                return tuple(int(n) for n in nums)
            return tuple(nums)
        except ValueError:
            return None

    # Single integer.
    try:
        return int(value_str)
    except ValueError:
        pass

    # Single float.
    try:
        return float(value_str)
    except ValueError:
        pass

    return None


def _sympify_eval_number_theory(
    nl_statement: str, n_val: int
) -> Optional[bool]:
    """Attempt to evaluate a number-theory conjecture at a single integer n.

    Uses heuristic SymPy sympification.  Returns ``None`` when the statement
    cannot be parsed or evaluated, preserving the fail-safe principle.

    Args:
        nl_statement: Natural-language conjecture statement.
        n_val: The integer value to substitute for ``n``.

    Returns:
        ``True`` if the statement appears to hold at ``n_val``, ``False`` if
        it is violated, or ``None`` if evaluation is inconclusive.
    """
    import sympy  # local import: SymPy is CPU-bound; keep imports lazy

    stmt = nl_statement.lower()
    n_sym = sympy.Symbol("n", positive=True, integer=True)

    # Divisibility: "k divides f(n)" or "f(n) divisible by k"
    div_patterns = [
        r"(\d+)\s+divides\s+(.+?)(?:\s+for|\s+when|\s*$)",
        r"(\d+)\s*\|\s*(.+?)(?:\s+for|\s+when|\s*$)",
        r"(.+?)\s+is\s+divisible\s+by\s+(\d+)",
    ]
    for pat in div_patterns:
        m = re.search(pat, stmt)
        if m is None:
            continue
        groups = m.groups()
        if "divisible by" in pat:
            expr_str, k_str = groups[0], groups[1]
        else:
            k_str, expr_str = groups[0], groups[1]
        try:
            k = int(k_str.strip())
            expr = sympy.sympify(
                expr_str.strip().rstrip("."),
                locals={"n": n_sym, "factorial": sympy.factorial},
            )
            result = int(expr.subs(n_sym, n_val).evalf())
            return result % k == 0
        except (ValueError, sympy.SympifyError, TypeError):
            continue

    # Modular: "f(n) mod m = c"
    mod_patterns = [
        r"(.+?)\s*(?:mod|%)\s*(\d+)\s*=\s*(\d+)",
        r"(.+?)\s+modulo\s+(\d+)\s+(?:is|equals?)\s+(\d+)",
    ]
    for pat in mod_patterns:
        m = re.search(pat, stmt)
        if m is None:
            continue
        try:
            expr = sympy.sympify(
                m.group(1).strip(),
                locals={"n": n_sym},
            )
            mod_k = int(m.group(2))
            rem = int(m.group(3))
            result = int(expr.subs(n_sym, n_val).evalf())
            return result % mod_k == rem
        except (ValueError, sympy.SympifyError, TypeError):
            continue

    # Primality: "EXPR is prime" — capture only the algebraic expression part.
    # The expression contains only math chars: n, digits, ^, +, -, *, /, (, ), spaces.
    prime_pat = re.search(
        r"([n\d\^\+\-\*\/\(\)\s]+)\s+is\s+(?:always\s+)?(?:a\s+)?prime(?:\s+number)?",
        stmt,
    )
    if prime_pat is not None:
        expr_str = prime_pat.group(1).strip().rstrip(".,").replace("^", "**")
        try:
            expr = sympy.sympify(
                expr_str,
                locals={"n": n_sym, "factorial": sympy.factorial},
            )
            val = int(expr.subs(n_sym, n_val).evalf())
            return bool(sympy.isprime(val))
        except (ValueError, sympy.SympifyError, TypeError):
            pass

    return None


def _sympify_eval_inequality(
    nl_statement: str, a_val: float, b_val: float, c_val: float
) -> Optional[bool]:
    """Evaluate an inequality conjecture at a specific (a, b, c) point.

    Args:
        nl_statement: Natural-language inequality statement.
        a_val: Value of variable ``a``.
        b_val: Value of variable ``b``.
        c_val: Value of variable ``c``.

    Returns:
        ``True`` if the inequality holds, ``False`` if violated, or ``None``
        if the statement cannot be parsed.
    """
    import sympy  # local import

    stmt_lower = nl_statement.lower()
    a_sym = sympy.Symbol("a", positive=True, real=True)
    b_sym = sympy.Symbol("b", positive=True, real=True)
    c_sym = sympy.Symbol("c", positive=True, real=True)
    sym_locals = {"a": a_sym, "b": b_sym, "c": c_sym}

    ineq_patterns = [
        (r"(.+?)\s*>=\s*(.+?)(?:\s+for|\s*$)", ">="),
        (r"(.+?)\s*<=\s*(.+?)(?:\s+for|\s*$)", "<="),
        (r"(.+?)\s*>\s*(.+?)(?:\s+for|\s*$)", ">"),
        (r"(.+?)\s*<\s*(.+?)(?:\s+for|\s*$)", "<"),
    ]
    eps = 1e-9
    for pat, op in ineq_patterns:
        m = re.search(pat, stmt_lower)
        if m is None:
            continue
        lhs_str = m.group(1).strip().rstrip(":")
        rhs_str = m.group(2).strip().rstrip(".")
        try:
            lhs = sympy.sympify(lhs_str, locals=sym_locals)
            rhs = sympy.sympify(rhs_str, locals=sym_locals)
            subs_map = {a_sym: a_val, b_sym: b_val, c_sym: c_val}
            lhs_val = float(lhs.subs(subs_map).evalf())
            rhs_val = float(rhs.subs(subs_map).evalf())
            if math.isnan(lhs_val) or math.isnan(rhs_val):
                return None
            if op == ">=":
                return lhs_val >= rhs_val - eps
            elif op == "<=":
                return lhs_val <= rhs_val + eps
            elif op == ">":
                return lhs_val > rhs_val - eps
            elif op == "<":
                return lhs_val < rhs_val + eps
        except (sympy.SympifyError, TypeError, ValueError, ZeroDivisionError):
            continue
    return None


def _sympify_eval_combinatorics(
    nl_statement: str, n_val: int
) -> Optional[bool]:
    """Evaluate a combinatorics conjecture at a specific integer n.

    Handles binomial identity patterns of the form C(f(n), g(n)) = h(n).

    Args:
        nl_statement: Natural-language combinatorics statement.
        n_val: The value of ``n`` to test.

    Returns:
        ``True`` if the claim holds, ``False`` if violated, or ``None`` if
        the statement cannot be parsed.
    """
    import sympy  # local import

    stmt_lower = nl_statement.lower()
    n_sym = sympy.Symbol("n", nonnegative=True, integer=True)

    binom_pattern = r"c\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*=\s*(.+?)(?:\s+for|\s*$)"
    m = re.search(binom_pattern, stmt_lower)
    if m is None:
        return None
    try:
        lhs_n = sympy.sympify(m.group(1).strip(), locals={"n": n_sym})
        lhs_k = sympy.sympify(m.group(2).strip(), locals={"n": n_sym})
        rhs = sympy.sympify(m.group(3).strip(), locals={"n": n_sym})
        binom_val = sympy.binomial(
            int(lhs_n.subs(n_sym, n_val)),
            int(lhs_k.subs(n_sym, n_val)),
        )
        rhs_val = int(rhs.subs(n_sym, n_val).evalf())
        return int(binom_val) == rhs_val
    except (sympy.SympifyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Refuter
# ---------------------------------------------------------------------------


class Refuter:
    """R-Agent: multi-strategy counterexample search for mathematical conjectures.

    The Refuter is the core contribution of the REFUTE paper.  It exposes four
    orthogonal search strategies that together cover the main failure modes of
    mathematical conjectures:

    * **BOUNDARY** — deterministic edge-case probing without LLM calls.
    * **RANDOM_STRUCTURED** — domain-aware random sampling without LLM calls.
    * **ANALOGICAL** — LLM-guided transfer from past successful refutations.
    * **SYMBOLIC_PERTURBATION** — LLM-guided parameter perturbation search.

    Each strategy is implemented as a private method and returns a
    ``CounterexampleCandidate`` (or ``None``) per candidate.  The public
    ``search`` and ``search_all_strategies`` methods orchestrate multiple
    rounds, collect all candidates, and propagate ``strategy_scores`` for
    S-Agent consumption.

    Attributes:
        client: Async LLM client used by the analogy and perturbation strategies.
        config: Pipeline configuration (temperature, batch sizes, etc.).
        max_concurrent: Maximum number of concurrent LLM coroutines allowed.
        _semaphore: Asyncio semaphore enforcing ``max_concurrent``.
    """

    def __init__(
        self,
        client: LLMClient,
        config: ConjLeanConfig,
        max_concurrent: int = 3,
    ) -> None:
        """Initialize the Refuter with an LLM client and pipeline configuration.

        Args:
            client: A concrete ``LLMClient`` instance (Anthropic, OpenAI, etc.).
            config: Fully validated ``ConjLeanConfig``.
            max_concurrent: Maximum simultaneous LLM calls.  Defaults to 3.

        Raises:
            TypeError: If ``client`` is not a ``LLMClient`` instance.
            TypeError: If ``config`` is not a ``ConjLeanConfig`` instance.
            ValueError: If ``max_concurrent`` is less than 1.
        """
        if not (hasattr(client, "complete") and hasattr(client, "complete_batch")):
            raise TypeError(
                f"client must implement complete() and complete_batch(), "
                f"got {type(client).__name__}"
            )
        if not isinstance(config, ConjLeanConfig):
            raise TypeError(
                f"config must be ConjLeanConfig, got {type(config).__name__}"
            )
        if max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be >= 1, got {max_concurrent}"
            )

        self.client = client
        self.config = config
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            "Refuter initialized | max_concurrent=%d",
            max_concurrent,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        conjecture: Conjecture,
        strategy: RefuterStrategy,
        max_rounds: int = 5,
        past_refutations: Optional[list[CounterexampleCandidate]] = None,
    ) -> RefuterResult:
        """Search for a counterexample using a single specified strategy.

        Runs up to ``max_rounds`` candidate generation rounds.  Each round
        produces one or more ``CounterexampleCandidate`` objects which are
        numerically verified inline.  The search halts early when a CONFIRMED
        counterexample is found.

        Args:
            conjecture: The conjecture to attempt to refute.
            strategy: Which of the four strategies to apply.
            max_rounds: Maximum number of generation rounds.  Defaults to 5.
            past_refutations: Historical confirmed candidates for ANALOGICAL
                and SYMBOLIC_PERTURBATION strategies.  Pass ``None`` or an
                empty list when no history is available.

        Returns:
            A ``RefuterResult`` summarising all candidates explored, the best
            confirmed counterexample (if any), and per-strategy success counts.

        Raises:
            TypeError: If ``conjecture`` is not a ``Conjecture`` instance.
            ValueError: If ``max_rounds`` is less than 1.
        """
        if not isinstance(conjecture, Conjecture):
            raise TypeError(
                f"conjecture must be a Conjecture, got {type(conjecture).__name__}"
            )
        if not conjecture.nl_statement.strip():
            raise ValueError(
                f"conjecture.nl_statement must not be empty (id={conjecture.id!r})"
            )
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")

        past_refutations = past_refutations or []
        result = RefuterResult(
            conjecture=conjecture,
            strategy_scores={s.value: 0 for s in RefuterStrategy},
        )

        logger.info(
            "Refuter.search | conjecture=%s | strategy=%s | max_rounds=%d",
            conjecture.id,
            strategy.value,
            max_rounds,
        )

        with tqdm(
            total=max_rounds,
            desc=f"R-Agent [{strategy.value}] {conjecture.id[:16]}",
            unit="round",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for round_idx in range(max_rounds):
                candidates = await self._run_strategy_round(
                    conjecture=conjecture,
                    strategy=strategy,
                    round_idx=round_idx,
                    past_refutations=past_refutations,
                )

                result.candidates.extend(candidates)
                result.rounds = round_idx + 1

                confirmed = next(
                    (c for c in candidates
                     if c.status == CounterexampleStatus.CONFIRMED),
                    None,
                )
                if confirmed is not None:
                    result.best_counterexample = confirmed
                    result.strategy_used = strategy
                    result.strategy_scores[strategy.value] += 1
                    pbar.set_postfix(status="REFUTED", refresh=False)
                    pbar.update(1)
                    logger.info(
                        "Counterexample confirmed | conjecture=%s | strategy=%s | "
                        "candidate=%s",
                        conjecture.id,
                        strategy.value,
                        confirmed.candidate_str[:80],
                    )
                    break

                pbar.set_postfix(
                    candidates=len(result.candidates),
                    status="searching",
                    refresh=False,
                )
                pbar.update(1)
                gc.collect()

        # Always record which strategy was run so callers can inspect it
        # even when no counterexample was found.
        if result.strategy_used is None:
            result.strategy_used = strategy

        logger.info(
            "Refuter.search complete | conjecture=%s | strategy=%s | "
            "rounds=%d | candidates=%d | refuted=%s",
            conjecture.id,
            strategy.value,
            result.rounds,
            len(result.candidates),
            result.best_counterexample is not None,
        )
        return result

    async def search_all_strategies(
        self,
        conjecture: Conjecture,
        max_rounds_per_strategy: int = 3,
        past_refutations: Optional[list[CounterexampleCandidate]] = None,
    ) -> RefuterResult:
        """Search for a counterexample across all four strategies sequentially.

        Strategies are tried in priority order:
        BOUNDARY → RANDOM_STRUCTURED → ANALOGICAL → SYMBOLIC_PERTURBATION.

        The search halts as soon as any strategy confirms a counterexample,
        avoiding unnecessary LLM spend.  All candidates from every strategy
        are accumulated in the returned result so the S-Agent has full
        trajectory information.

        Args:
            conjecture: The conjecture to attempt to refute.
            max_rounds_per_strategy: Rounds budget given to each strategy.
                Defaults to 3.
            past_refutations: Historical confirmed candidates used by ANALOGICAL
                and SYMBOLIC_PERTURBATION strategies.

        Returns:
            A merged ``RefuterResult`` containing candidates from all strategies
            tried, the best confirmed counterexample, and aggregated
            ``strategy_scores``.

        Raises:
            TypeError: If ``conjecture`` is not a ``Conjecture`` instance.
            ValueError: If ``max_rounds_per_strategy`` is less than 1.
        """
        if not isinstance(conjecture, Conjecture):
            raise TypeError(
                f"conjecture must be a Conjecture, got {type(conjecture).__name__}"
            )
        if max_rounds_per_strategy < 1:
            raise ValueError(
                f"max_rounds_per_strategy must be >= 1, got {max_rounds_per_strategy}"
            )

        past_refutations = past_refutations or []
        strategy_order = [
            RefuterStrategy.BOUNDARY,
            RefuterStrategy.RANDOM_STRUCTURED,
            RefuterStrategy.ANALOGICAL,
            RefuterStrategy.SYMBOLIC_PERTURBATION,
        ]

        merged = RefuterResult(
            conjecture=conjecture,
            strategy_scores={s.value: 0 for s in RefuterStrategy},
        )

        logger.info(
            "Refuter.search_all_strategies | conjecture=%s | "
            "max_rounds_per_strategy=%d",
            conjecture.id,
            max_rounds_per_strategy,
        )

        for strategy in strategy_order:
            sub_result = await self.search(
                conjecture=conjecture,
                strategy=strategy,
                max_rounds=max_rounds_per_strategy,
                past_refutations=past_refutations,
            )

            merged.candidates.extend(sub_result.candidates)
            merged.rounds += sub_result.rounds
            for key, score in sub_result.strategy_scores.items():
                merged.strategy_scores[key] = (
                    merged.strategy_scores.get(key, 0) + score
                )

            if sub_result.best_counterexample is not None:
                merged.best_counterexample = sub_result.best_counterexample
                merged.strategy_used = strategy
                logger.info(
                    "search_all_strategies: confirmed on strategy=%s | conjecture=%s",
                    strategy.value,
                    conjecture.id,
                )
                break

        logger.info(
            "search_all_strategies complete | conjecture=%s | "
            "total_candidates=%d | total_rounds=%d | refuted=%s",
            conjecture.id,
            len(merged.candidates),
            merged.rounds,
            merged.best_counterexample is not None,
        )
        return merged

    # ------------------------------------------------------------------
    # Strategy dispatcher
    # ------------------------------------------------------------------

    async def _run_strategy_round(
        self,
        conjecture: Conjecture,
        strategy: RefuterStrategy,
        round_idx: int,
        past_refutations: list[CounterexampleCandidate],
    ) -> list[CounterexampleCandidate]:
        """Dispatch one round of candidate generation to the appropriate strategy.

        Args:
            conjecture: The conjecture being refuted.
            strategy: Which strategy to invoke.
            round_idx: Zero-based round index (used for seeding randomness).
            past_refutations: Historical confirmed counterexamples.

        Returns:
            List of ``CounterexampleCandidate`` objects generated this round,
            each with a ``status`` of either CONFIRMED or NOT_CONFIRMED.
        """
        if strategy == RefuterStrategy.BOUNDARY:
            return await self._boundary_probe(conjecture, conjecture.domain)
        elif strategy == RefuterStrategy.RANDOM_STRUCTURED:
            return await self._random_structured(
                conjecture, conjecture.domain, n_attempts=20 + round_idx * 10
            )
        elif strategy == RefuterStrategy.ANALOGICAL:
            return await self._analogical_transfer(conjecture, past_refutations)
        elif strategy == RefuterStrategy.SYMBOLIC_PERTURBATION:
            return await self._symbolic_perturbation(
                conjecture, n_perturbations=5 + round_idx * 2
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # BOUNDARY strategy
    # ------------------------------------------------------------------

    async def _boundary_probe(
        self,
        conjecture: Conjecture,
        domain: Domain,
    ) -> list[CounterexampleCandidate]:
        """Probe deterministic boundary / edge-case values for the given domain.

        No LLM calls are made; all evaluation is performed via SymPy in an
        async-safe worker thread.  The probe set is fixed and covers:

        - NUMBER_THEORY: ``n`` in ``{0, 1, 2, 3, 4, 5, 10, 100, 1000}`` plus
          the first 20 primes.
        - INEQUALITY: ``(a, b, c)`` at canonical extremes and unit / near-zero.
        - COMBINATORICS: ``n`` in ``{0, 1, ..., 25}``.

        Args:
            conjecture: The conjecture to probe.
            domain: Mathematical domain (determines the probe grid).

        Returns:
            List of ``CounterexampleCandidate`` objects with CONFIRMED status
            for any value that falsifies the conjecture, NOT_CONFIRMED otherwise.
        """
        candidates: list[CounterexampleCandidate] = []

        if domain == Domain.NUMBER_THEORY:
            probe_vals = (
                list(range(0, 50))  # covers small and medium boundary values
                + [100, 1000]
                + _FIRST_20_PRIMES
            )
            probe_vals = list(dict.fromkeys(probe_vals))  # deduplicate, preserve order

            for n_val in probe_vals:
                result = await asyncio.to_thread(
                    _sympify_eval_number_theory, conjecture.nl_statement, n_val
                )
                if result is False:
                    candidate = CounterexampleCandidate(
                        conjecture_id=conjecture.id,
                        candidate_str=f"n={n_val}",
                        strategy=RefuterStrategy.BOUNDARY,
                        status=CounterexampleStatus.CONFIRMED,
                        evidence={"n": n_val, "holds": False},
                        reasoning=f"Boundary probe: n={n_val} falsifies the statement.",
                    )
                    logger.debug(
                        "BOUNDARY confirmed counterexample | conjecture=%s | n=%d",
                        conjecture.id,
                        n_val,
                    )
                    candidates.append(candidate)
                    return candidates  # early-exit on first confirmed

        elif domain == Domain.INEQUALITY:
            probe_points: list[tuple[float, float, float]] = [
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                (0.001, 0.001, 0.001),
                (100.0, 100.0, 100.0),
                (1.0, 2.0, 3.0),
                (0.5, 1.0, 2.0),
                (1.0, 0.001, 1000.0),
                (10.0, 0.1, 5.0),
            ]
            for a_val, b_val, c_val in probe_points:
                if a_val == 0.0 and b_val == 0.0 and c_val == 0.0:
                    # (0,0,0) can cause domain errors; skip gracefully.
                    continue
                result = await asyncio.to_thread(
                    _sympify_eval_inequality,
                    conjecture.nl_statement,
                    a_val,
                    b_val,
                    c_val,
                )
                if result is False:
                    candidate = CounterexampleCandidate(
                        conjecture_id=conjecture.id,
                        candidate_str=(
                            f"a={a_val}, b={b_val}, c={c_val}"
                        ),
                        strategy=RefuterStrategy.BOUNDARY,
                        status=CounterexampleStatus.CONFIRMED,
                        evidence={"a": a_val, "b": b_val, "c": c_val, "holds": False},
                        reasoning=(
                            f"Boundary probe: (a={a_val}, b={b_val}, c={c_val}) "
                            "falsifies the inequality."
                        ),
                    )
                    logger.debug(
                        "BOUNDARY confirmed counterexample | conjecture=%s | "
                        "point=(%s,%s,%s)",
                        conjecture.id,
                        a_val,
                        b_val,
                        c_val,
                    )
                    candidates.append(candidate)
                    return candidates

        elif domain == Domain.COMBINATORICS:
            probe_vals = list(range(26))  # 0 through 25

            for n_val in probe_vals:
                result = await asyncio.to_thread(
                    _sympify_eval_combinatorics, conjecture.nl_statement, n_val
                )
                if result is False:
                    candidate = CounterexampleCandidate(
                        conjecture_id=conjecture.id,
                        candidate_str=f"n={n_val}",
                        strategy=RefuterStrategy.BOUNDARY,
                        status=CounterexampleStatus.CONFIRMED,
                        evidence={"n": n_val, "holds": False},
                        reasoning=f"Boundary probe: n={n_val} falsifies the statement.",
                    )
                    logger.debug(
                        "BOUNDARY confirmed counterexample | conjecture=%s | n=%d",
                        conjecture.id,
                        n_val,
                    )
                    candidates.append(candidate)
                    return candidates

        logger.debug(
            "BOUNDARY probe found no counterexample | conjecture=%s | domain=%s",
            conjecture.id,
            domain.value,
        )
        return candidates

    # ------------------------------------------------------------------
    # RANDOM_STRUCTURED strategy
    # ------------------------------------------------------------------

    async def _random_structured(
        self,
        conjecture: Conjecture,
        domain: Domain,
        n_attempts: int = 20,
    ) -> list[CounterexampleCandidate]:
        """Generate mathematically-structured random candidates without LLM calls.

        Candidates are drawn from domain-specific distributions designed to
        stress conjectures:

        - NUMBER_THEORY: random integers from the union of primes, squares,
          composites, Fibonacci numbers, and uniform random integers.
        - INEQUALITY: random positive reals with diverse ratio structures.
        - COMBINATORICS: uniform random integers in ``[0, 200]``.

        All verification is performed via SymPy in async worker threads.

        Args:
            conjecture: The conjecture to probe.
            domain: Mathematical domain controlling the sampling distribution.
            n_attempts: Number of random candidates to evaluate.  Defaults to 20.

        Returns:
            List of confirmed ``CounterexampleCandidate`` objects (at most one,
            since search halts on first confirmed).  Empty list when no
            counterexample is found.
        """
        rng = random.Random()  # not seeded: want genuine randomness across calls
        candidates: list[CounterexampleCandidate] = []

        if domain == Domain.NUMBER_THEORY:
            pool: list[int] = []

            # Primes up to ~500 via trial division.
            def _is_prime(k: int) -> bool:
                if k < 2:
                    return False
                if k == 2:
                    return True
                if k % 2 == 0:
                    return False
                for i in range(3, int(k ** 0.5) + 1, 2):
                    if k % i == 0:
                        return False
                return True

            primes_ext = [p for p in range(2, 500) if _is_prime(p)]
            squares = [i * i for i in range(1, 50)]
            composites = [k for k in range(4, 200) if not _is_prime(k)]
            fib = _FIBONACCI_NUMBERS[2:]  # skip 0, 1 (covered by boundary)
            uniform = [rng.randint(2, 10_000) for _ in range(40)]

            pool = primes_ext + squares + composites + fib + uniform
            sampled: list[int] = rng.sample(pool, min(n_attempts, len(pool)))

            for n_val in sampled:
                result = await asyncio.to_thread(
                    _sympify_eval_number_theory, conjecture.nl_statement, n_val
                )
                if result is False:
                    candidate = CounterexampleCandidate(
                        conjecture_id=conjecture.id,
                        candidate_str=f"n={n_val}",
                        strategy=RefuterStrategy.RANDOM_STRUCTURED,
                        status=CounterexampleStatus.CONFIRMED,
                        evidence={"n": n_val, "holds": False},
                        reasoning=(
                            f"Random structured probe: n={n_val} (from structured "
                            "number-theory pool) falsifies the statement."
                        ),
                    )
                    logger.debug(
                        "RANDOM_STRUCTURED confirmed | conjecture=%s | n=%d",
                        conjecture.id,
                        n_val,
                    )
                    candidates.append(candidate)
                    return candidates

        elif domain == Domain.INEQUALITY:
            def _sample_point() -> tuple[float, float, float]:
                """Sample a positive-real triple with structurally diverse ratios."""
                mode = rng.choice(["unit", "large", "small", "mixed", "extreme"])
                if mode == "unit":
                    return (
                        rng.uniform(0.5, 2.0),
                        rng.uniform(0.5, 2.0),
                        rng.uniform(0.5, 2.0),
                    )
                elif mode == "large":
                    return (
                        rng.uniform(10.0, 1000.0),
                        rng.uniform(10.0, 1000.0),
                        rng.uniform(10.0, 1000.0),
                    )
                elif mode == "small":
                    return (
                        rng.uniform(1e-4, 0.1),
                        rng.uniform(1e-4, 0.1),
                        rng.uniform(1e-4, 0.1),
                    )
                elif mode == "mixed":
                    return (
                        rng.uniform(1e-3, 1e3),
                        rng.uniform(1e-3, 1e3),
                        rng.uniform(1e-3, 1e3),
                    )
                else:  # extreme
                    return (
                        rng.choice([1e-6, 1e-4, 1e6, 1e8]),
                        rng.choice([1e-6, 1e-4, 1e6, 1e8]),
                        rng.choice([1e-6, 1e-4, 1e6, 1e8]),
                    )

            for _ in range(n_attempts):
                a_val, b_val, c_val = _sample_point()
                result = await asyncio.to_thread(
                    _sympify_eval_inequality,
                    conjecture.nl_statement,
                    a_val,
                    b_val,
                    c_val,
                )
                if result is False:
                    candidate = CounterexampleCandidate(
                        conjecture_id=conjecture.id,
                        candidate_str=f"a={a_val:.6g}, b={b_val:.6g}, c={c_val:.6g}",
                        strategy=RefuterStrategy.RANDOM_STRUCTURED,
                        status=CounterexampleStatus.CONFIRMED,
                        evidence={"a": a_val, "b": b_val, "c": c_val, "holds": False},
                        reasoning=(
                            f"Random structured probe: "
                            f"(a={a_val:.6g}, b={b_val:.6g}, c={c_val:.6g}) "
                            "falsifies the inequality."
                        ),
                    )
                    logger.debug(
                        "RANDOM_STRUCTURED confirmed | conjecture=%s | "
                        "point=(%.4g,%.4g,%.4g)",
                        conjecture.id,
                        a_val,
                        b_val,
                        c_val,
                    )
                    candidates.append(candidate)
                    return candidates

        elif domain == Domain.COMBINATORICS:
            sampled_n = [rng.randint(0, 200) for _ in range(n_attempts)]
            for n_val in sampled_n:
                result = await asyncio.to_thread(
                    _sympify_eval_combinatorics, conjecture.nl_statement, n_val
                )
                if result is False:
                    candidate = CounterexampleCandidate(
                        conjecture_id=conjecture.id,
                        candidate_str=f"n={n_val}",
                        strategy=RefuterStrategy.RANDOM_STRUCTURED,
                        status=CounterexampleStatus.CONFIRMED,
                        evidence={"n": n_val, "holds": False},
                        reasoning=(
                            f"Random structured probe: n={n_val} falsifies the "
                            "combinatorics statement."
                        ),
                    )
                    logger.debug(
                        "RANDOM_STRUCTURED confirmed | conjecture=%s | n=%d",
                        conjecture.id,
                        n_val,
                    )
                    candidates.append(candidate)
                    return candidates

        logger.debug(
            "RANDOM_STRUCTURED found no counterexample | conjecture=%s | "
            "domain=%s | n_attempts=%d",
            conjecture.id,
            domain.value,
            n_attempts,
        )
        return candidates

    # ------------------------------------------------------------------
    # ANALOGICAL strategy
    # ------------------------------------------------------------------

    async def _analogical_transfer(
        self,
        conjecture: Conjecture,
        past_refutations: list[CounterexampleCandidate],
    ) -> list[CounterexampleCandidate]:
        """Transfer counterexample patterns from past refutations via LLM reasoning.

        Constructs a prompt describing the target conjecture and up to 10 past
        confirmed counterexamples for similar conjectures.  The LLM is asked to
        propose new candidate values inspired by the analogical patterns it
        observes.  Each suggestion is then verified numerically via SymPy.

        If no past refutations are available, the method falls through to an
        LLM cold-start prompt asking for promising values based on mathematical
        intuition alone.

        Args:
            conjecture: The conjecture to attempt to refute.
            past_refutations: Confirmed counterexample candidates from previous
                refutations of other or earlier conjectures.

        Returns:
            List of verified ``CounterexampleCandidate`` objects.  Contains at
            most one CONFIRMED entry (search halts on confirmation); may contain
            multiple NOT_CONFIRMED entries.
        """
        recent_examples = past_refutations[-10:]  # cap context window

        examples_block: str
        if recent_examples:
            example_lines = "\n".join(
                f"  - Conjecture snippet: '{ex.conjecture_id}' | "
                f"Counterexample: {ex.candidate_str} | "
                f"Reasoning: {ex.reasoning[:120]}"
                for ex in recent_examples
            )
            examples_block = (
                "Here are past successful counterexamples for similar conjectures "
                "(ordered most-recent first):\n" + example_lines
            )
        else:
            examples_block = (
                "No past counterexamples are available.  Use mathematical intuition "
                "to propose promising test values."
            )

        domain_hint = (
            "integers n" if conjecture.domain == Domain.NUMBER_THEORY
            else "real triples (a, b, c)" if conjecture.domain == Domain.INEQUALITY
            else "integers n"
        )

        prompt = (
            f"You are an expert mathematical falsifier.\n\n"
            f"Target conjecture ({conjecture.domain.value}):\n"
            f"  \"{conjecture.nl_statement}\"\n\n"
            f"{examples_block}\n\n"
            f"Based on the above patterns, propose 5 specific {domain_hint} that "
            f"are most likely to falsify the target conjecture.  Respond ONLY with "
            f"valid JSON in this exact schema:\n"
            f'{{"candidates": [{{"value": "<value>", "reasoning": "<brief reason>"}}]}}\n'
            f"No extra text outside the JSON block."
        )

        messages = [
            {"role": "system", "content": "You propose mathematical counterexample candidates."},
            {"role": "user", "content": prompt},
        ]

        async with self._semaphore:
            raw_response = await self.client.complete(
                messages=messages,
                temperature=_LLM_TEMPERATURE,
                max_tokens=_LLM_MAX_TOKENS,
            )

        logger.debug(
            "ANALOGICAL LLM response | conjecture=%s | response_len=%d",
            conjecture.id,
            len(raw_response),
        )

        parsed = _parse_llm_candidates(raw_response)
        return await self._verify_candidates(
            conjecture=conjecture,
            raw_candidates=parsed,
            strategy=RefuterStrategy.ANALOGICAL,
        )

    # ------------------------------------------------------------------
    # SYMBOLIC_PERTURBATION strategy
    # ------------------------------------------------------------------

    async def _symbolic_perturbation(
        self,
        conjecture: Conjecture,
        n_perturbations: int = 5,
    ) -> list[CounterexampleCandidate]:
        """Identify critical parameters and probe perturbations via LLM + SymPy.

        Two LLM calls are made in sequence:

        1. **Parameter identification**: Ask the LLM which constants, exponents,
           or thresholds in the conjecture are "critical" — i.e., small changes
           would likely break the statement.
        2. **Perturbation generation**: Ask the LLM to propose concrete perturbed
           parameter values (weakened constants, flipped inequalities, changed
           exponents) that would make the statement false.

        Each proposed perturbation is recorded as a candidate and verified
        numerically.  For statements where SymPy cannot parse the perturbed
        form, the candidate is recorded as UNCERTAIN with the LLM's reasoning
        as evidence.

        Args:
            conjecture: The conjecture to perturb.
            n_perturbations: Number of perturbation candidates to request from
                the LLM.  Defaults to 5.

        Returns:
            List of ``CounterexampleCandidate`` objects.  Confirmed entries
            indicate numeric falsification; UNCERTAIN entries indicate LLM
            assessed the perturbation as boundary evidence even if SymPy
            could not verify.
        """
        # --- Phase 1: identify critical parameters ---
        id_prompt = (
            f"You are an expert mathematical analyst.\n\n"
            f"Conjecture ({conjecture.domain.value}):\n"
            f"  \"{conjecture.nl_statement}\"\n\n"
            f"Identify the 1–3 most critical numerical parameters, constants, or "
            f"structural features of this statement (e.g., exponents, divisors, "
            f"inequality bounds, thresholds).  List them concisely, one per line."
        )

        async with self._semaphore:
            critical_params_text = await self.client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "You identify critical parameters in mathematical statements.",
                    },
                    {"role": "user", "content": id_prompt},
                ],
                temperature=0.3,
                max_tokens=256,
            )

        logger.debug(
            "SYMBOLIC_PERTURBATION critical params | conjecture=%s | params=%s",
            conjecture.id,
            critical_params_text[:120],
        )

        # --- Phase 2: generate perturbation candidates ---
        domain_hint = (
            "integer n" if conjecture.domain == Domain.NUMBER_THEORY
            else "real triple (a, b, c)" if conjecture.domain == Domain.INEQUALITY
            else "integer n"
        )

        perturb_prompt = (
            f"You are an expert mathematical falsifier.\n\n"
            f"Conjecture ({conjecture.domain.value}):\n"
            f"  \"{conjecture.nl_statement}\"\n\n"
            f"Critical parameters identified:\n{critical_params_text.strip()}\n\n"
            f"Propose {n_perturbations} specific {domain_hint} values that exploit "
            f"boundary conditions near the critical parameters — i.e., values at "
            f"which the statement is most likely to fail.  For each candidate, "
            f"briefly explain why it is near a boundary.\n\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"candidates": [{{"value": "<value>", "reasoning": "<brief reason>"}}]}}\n'
            f"No extra text outside the JSON block."
        )

        async with self._semaphore:
            raw_response = await self.client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "You propose boundary counterexample candidates.",
                    },
                    {"role": "user", "content": perturb_prompt},
                ],
                temperature=_LLM_TEMPERATURE,
                max_tokens=_LLM_MAX_TOKENS,
            )

        logger.debug(
            "SYMBOLIC_PERTURBATION LLM response | conjecture=%s | response_len=%d",
            conjecture.id,
            len(raw_response),
        )

        parsed = _parse_llm_candidates(raw_response)
        return await self._verify_candidates(
            conjecture=conjecture,
            raw_candidates=parsed,
            strategy=RefuterStrategy.SYMBOLIC_PERTURBATION,
        )

    # ------------------------------------------------------------------
    # Shared numeric verification
    # ------------------------------------------------------------------

    async def _verify_candidates(
        self,
        conjecture: Conjecture,
        raw_candidates: list[dict[str, Any]],
        strategy: RefuterStrategy,
    ) -> list[CounterexampleCandidate]:
        """Numerically verify a list of raw LLM-proposed candidates via SymPy.

        For each candidate dict (must have ``"value"`` key), the numeric value
        is extracted and the appropriate domain evaluator is called in a worker
        thread.  Candidates that SymPy cannot parse are recorded as UNCERTAIN.

        Args:
            conjecture: The conjecture being falsified.
            raw_candidates: List of dicts from ``_parse_llm_candidates``.
            strategy: The strategy that produced these candidates (for
                attribution in returned objects).

        Returns:
            List of ``CounterexampleCandidate`` objects with statuses populated.
            The list may be empty if ``raw_candidates`` is empty or if all
            values fail numeric extraction.
        """
        verified: list[CounterexampleCandidate] = []
        domain = conjecture.domain

        for raw in raw_candidates:
            value_str = str(raw.get("value", "")).strip()
            reasoning = str(raw.get("reasoning", "")).strip()
            numeric = _extract_numeric_value(value_str)

            if numeric is None:
                logger.debug(
                    "Could not extract numeric value from candidate=%r | conjecture=%s",
                    value_str,
                    conjecture.id,
                )
                verified.append(
                    CounterexampleCandidate(
                        conjecture_id=conjecture.id,
                        candidate_str=value_str,
                        strategy=strategy,
                        status=CounterexampleStatus.UNCERTAIN,
                        evidence={"raw_value": value_str},
                        reasoning=reasoning,
                    )
                )
                continue

            holds: Optional[bool] = None

            if domain == Domain.NUMBER_THEORY:
                n_val = int(numeric) if not isinstance(numeric, tuple) else None
                if n_val is not None and n_val >= 0:
                    holds = await asyncio.to_thread(
                        _sympify_eval_number_theory, conjecture.nl_statement, n_val
                    )

            elif domain == Domain.INEQUALITY:
                if isinstance(numeric, tuple) and len(numeric) >= 3:
                    a_val, b_val, c_val = float(numeric[0]), float(numeric[1]), float(numeric[2])
                elif isinstance(numeric, (int, float)):
                    a_val = b_val = c_val = float(numeric)
                else:
                    a_val = b_val = c_val = None  # type: ignore[assignment]

                if a_val is not None:
                    holds = await asyncio.to_thread(
                        _sympify_eval_inequality,
                        conjecture.nl_statement,
                        a_val,
                        b_val,
                        c_val,
                    )

            elif domain == Domain.COMBINATORICS:
                n_val = int(numeric) if not isinstance(numeric, tuple) else None
                if n_val is not None and n_val >= 0:
                    holds = await asyncio.to_thread(
                        _sympify_eval_combinatorics, conjecture.nl_statement, n_val
                    )

            if holds is False:
                status = CounterexampleStatus.CONFIRMED
                logger.debug(
                    "Verified CONFIRMED counterexample | conjecture=%s | "
                    "strategy=%s | value=%r",
                    conjecture.id,
                    strategy.value,
                    value_str,
                )
            elif holds is True:
                status = CounterexampleStatus.NOT_CONFIRMED
            else:
                status = CounterexampleStatus.UNCERTAIN

            verified.append(
                CounterexampleCandidate(
                    conjecture_id=conjecture.id,
                    candidate_str=value_str,
                    strategy=strategy,
                    status=status,
                    evidence={"raw_value": value_str, "numeric": numeric, "holds": holds},
                    reasoning=reasoning,
                )
            )

            if status == CounterexampleStatus.CONFIRMED:
                # Halt verification on first confirmed candidate to save compute.
                return verified

        return verified
