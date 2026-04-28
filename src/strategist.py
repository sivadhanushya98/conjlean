"""
S-Agent: Strategist meta-controller for the REFUTE pipeline.

The Strategist is responsible for selecting the next R-Agent search strategy,
maintaining cross-conjecture strategy success statistics, deciding when the
refutation budget is exhausted, and producing LLM-generated analyses of failure
patterns for the research notebook.

Strategy selection follows a UCB1-style exploration/exploitation policy over
the four available RefuterStrategy values, augmented by domain-specific
success statistics and an LLM fallback when per-strategy sample counts are
too low to trust the empirical estimates.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from conjlean.schemas import (
    Conjecture,
    Domain,
    RefuterResult,
    RefuterStrategy,
)

if TYPE_CHECKING:
    from conjlean.config import ConjLeanConfig
    from conjlean.models import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UCB1_EXPLORATION_CONSTANT: float = 1.414  # sqrt(2)
_MIN_SAMPLES_FOR_UCB: int = 5          # below this, ask the LLM for guidance
_STOP_CONFIDENCE_THRESHOLD: float = 0.05  # if best strategy win-rate < 5 %, stop
_LLM_STRATEGY_TEMPERATURE: float = 0.2
_LLM_ANALYSIS_TEMPERATURE: float = 0.4
_MAX_TOKENS_STRATEGY: int = 256
_MAX_TOKENS_ANALYSIS: int = 1024

# Canonical ordering: cheapest first, so the default sequential fallback is sane
_STRATEGY_PRIORITY_ORDER: list[RefuterStrategy] = [
    RefuterStrategy.BOUNDARY,
    RefuterStrategy.RANDOM_STRUCTURED,
    RefuterStrategy.SYMBOLIC_PERTURBATION,
    RefuterStrategy.ANALOGICAL,
]

_STRATEGY_DISPLAY_NAMES: dict[RefuterStrategy, str] = {
    RefuterStrategy.BOUNDARY: "boundary",
    RefuterStrategy.RANDOM_STRUCTURED: "random_structured",
    RefuterStrategy.SYMBOLIC_PERTURBATION: "symbolic_perturbation",
    RefuterStrategy.ANALOGICAL: "analogical",
}


# ---------------------------------------------------------------------------
# Internal stats container
# ---------------------------------------------------------------------------


@dataclass
class _StrategyStats:
    """Per-strategy rolling statistics maintained by the Strategist.

    Attributes:
        attempts: Total number of times this strategy was tried.
        successes: Number of attempts that produced a confirmed counterexample.
        domain_attempts: Breakdown of attempts by domain.
        domain_successes: Breakdown of successes by domain.
    """

    attempts: int = 0
    successes: int = 0
    domain_attempts: dict[str, int] = field(default_factory=dict)
    domain_successes: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Empirical success rate (0.0 when no attempts have been made)."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def domain_win_rate(self, domain: Domain) -> float:
        """Empirical success rate for a specific domain.

        Args:
            domain: The target mathematical domain.

        Returns:
            Per-domain win rate, 0.0 when no domain-specific attempts exist.
        """
        d_key = domain.value
        d_attempts = self.domain_attempts.get(d_key, 0)
        if d_attempts == 0:
            return 0.0
        return self.domain_successes.get(d_key, 0) / d_attempts


# ---------------------------------------------------------------------------
# Strategist
# ---------------------------------------------------------------------------


class Strategist:
    """S-Agent meta-controller that manages R-Agent strategy selection.

    The Strategist orchestrates strategy allocation for refutation attempts
    using a UCB1 bandit policy augmented by domain-aware statistics and LLM
    guidance when empirical data is insufficient.  It also maintains a
    per-run decision log for research reproducibility and supports an LLM-
    generated failure-pattern analysis for the paper's research notebook.

    Strategy selection priority:
    1. If no strategies have been tried: start with BOUNDARY (cheapest, no LLM).
    2. If BOUNDARY failed and RANDOM_STRUCTURED is untried: escalate.
    3. If sufficient samples exist (>= _MIN_SAMPLES_FOR_UCB per strategy):
       use UCB1 over untried strategies, preferring domain-successful ones.
    4. If samples are insufficient: ask the LLM for a recommendation.
    5. Analogical transfer is preferred when the domain matches past successes.

    Attributes:
        client: The LLMClient used for strategy recommendation and analysis.
        config: Pipeline configuration object.
        _global_stats: Per-strategy aggregate statistics across all conjectures.
        _decision_log: Ordered list of strategy decisions (for research logging).
    """

    def __init__(self, client: "LLMClient", config: "ConjLeanConfig") -> None:
        """Initialise the Strategist with LLM client and pipeline configuration.

        Args:
            client: An async LLMClient for LLM-guided strategy selection.
            config: A validated ConjLeanConfig instance.
        """
        self.client = client
        self.config = config

        self._global_stats: dict[RefuterStrategy, _StrategyStats] = {
            strategy: _StrategyStats() for strategy in RefuterStrategy
        }
        self._decision_log: list[dict] = []

        logger.info(
            "Strategist initialised | strategies=%d | ucb_exploration=%.3f",
            len(RefuterStrategy),
            _UCB1_EXPLORATION_CONSTANT,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select_strategy(
        self,
        conjecture: Conjecture,
        past_results: list[RefuterResult],
        tried_strategies: set[RefuterStrategy],
        global_strategy_stats: dict[str, dict],
    ) -> RefuterStrategy:
        """Select the next R-Agent strategy for a given conjecture.

        Evaluates all un-tried strategies and ranks them via UCB1 plus domain-
        aware bonuses.  Falls back to LLM recommendation when global statistics
        are too sparse to trust.

        Args:
            conjecture: The conjecture being refuted.
            past_results: R-Agent results from previous rounds on this
                conjecture (empty on round 0).
            tried_strategies: Strategies already attempted for this conjecture
                in the current loop.
            global_strategy_stats: Cross-conjecture strategy performance dict
                with structure ``{strategy_name: {"attempts": N, "successes": M}}``.
                May be empty on the first conjecture.

        Returns:
            The selected RefuterStrategy for the next round.

        Raises:
            ValueError: If all strategies have already been tried.
        """
        available: list[RefuterStrategy] = [
            s for s in _STRATEGY_PRIORITY_ORDER if s not in tried_strategies
        ]

        if not available:
            raise ValueError(
                f"All {len(RefuterStrategy)} strategies have been tried for "
                f"conjecture {conjecture.id!r}. Cannot select a new strategy."
            )

        # ── Rule 1: first attempt → always start with BOUNDARY ──────────
        if not tried_strategies:
            chosen = RefuterStrategy.BOUNDARY
            self._log_decision(
                conjecture_id=conjecture.id,
                chosen=chosen,
                reason="first_attempt_boundary_default",
            )
            logger.debug(
                "Strategist | conjecture=%s | selected=BOUNDARY | reason=first_attempt",
                conjecture.id,
            )
            return chosen

        # ── Rule 2: BOUNDARY failed, RANDOM_STRUCTURED not tried ────────
        if (
            RefuterStrategy.BOUNDARY in tried_strategies
            and RefuterStrategy.RANDOM_STRUCTURED not in tried_strategies
        ):
            chosen = RefuterStrategy.RANDOM_STRUCTURED
            self._log_decision(
                conjecture_id=conjecture.id,
                chosen=chosen,
                reason="boundary_failed_escalate_random_structured",
            )
            logger.debug(
                "Strategist | conjecture=%s | selected=RANDOM_STRUCTURED | reason=boundary_escalation",
                conjecture.id,
            )
            return chosen

        # ── Rule 3: Check if ANALOGICAL has domain-matched past success ──
        if RefuterStrategy.ANALOGICAL in available:
            if self._has_domain_analogical_success(conjecture.domain):
                chosen = RefuterStrategy.ANALOGICAL
                self._log_decision(
                    conjecture_id=conjecture.id,
                    chosen=chosen,
                    reason="domain_analogical_transfer_preferred",
                )
                logger.debug(
                    "Strategist | conjecture=%s | selected=ANALOGICAL | reason=domain_match",
                    conjecture.id,
                )
                return chosen

        # ── Rule 4: Determine if stats are sufficient for UCB1 ───────────
        all_stats_sufficient = all(
            self._global_stats[s].attempts >= _MIN_SAMPLES_FOR_UCB
            for s in available
        )

        if not all_stats_sufficient:
            # Ask the LLM for guidance when empirical data is thin
            chosen = await self._llm_select_strategy(
                conjecture=conjecture,
                available=available,
                past_results=past_results,
                global_strategy_stats=global_strategy_stats,
            )
            self._log_decision(
                conjecture_id=conjecture.id,
                chosen=chosen,
                reason="llm_guided_sparse_stats",
            )
            logger.debug(
                "Strategist | conjecture=%s | selected=%s | reason=llm_guided",
                conjecture.id,
                chosen.value,
            )
            return chosen

        # ── Rule 5: UCB1 selection over available strategies ─────────────
        chosen = self._ucb1_select(
            available=available,
            domain=conjecture.domain,
        )
        self._log_decision(
            conjecture_id=conjecture.id,
            chosen=chosen,
            reason="ucb1_exploitation_exploration",
        )
        logger.debug(
            "Strategist | conjecture=%s | selected=%s | reason=ucb1",
            conjecture.id,
            chosen.value,
        )
        return chosen

    def update_stats(
        self,
        strategy: RefuterStrategy,
        success: bool,
        domain: Domain,
    ) -> None:
        """Update rolling success statistics after an R-Agent round completes.

        Should be called once per completed RefuterResult, regardless of
        outcome, to keep statistics current for subsequent selections.

        Args:
            strategy: The strategy that was used in the completed round.
            success: True if a confirmed counterexample was produced.
            domain: The mathematical domain of the conjecture that was attempted.
        """
        stats = self._global_stats[strategy]
        stats.attempts += 1
        d_key = domain.value
        stats.domain_attempts[d_key] = stats.domain_attempts.get(d_key, 0) + 1

        if success:
            stats.successes += 1
            stats.domain_successes[d_key] = stats.domain_successes.get(d_key, 0) + 1

        logger.debug(
            "Strategist.update_stats | strategy=%s | domain=%s | success=%s | "
            "total_attempts=%d | total_successes=%d | win_rate=%.3f",
            strategy.value,
            domain.value,
            success,
            stats.attempts,
            stats.successes,
            stats.win_rate,
        )

    def should_stop(
        self,
        conjecture: Conjecture,
        rounds_used: int,
        max_rounds: int,
        results_so_far: list[RefuterResult],
    ) -> tuple[bool, str]:
        """Decide whether the REFUTE loop should terminate early for a conjecture.

        Evaluates multiple stopping conditions in priority order:
        1. All four strategies have been tried at least once.
        2. Round budget is exhausted.
        3. Best available strategy win-rate is below confidence threshold.

        Args:
            conjecture: The conjecture currently being refuted.
            rounds_used: Number of R-Agent rounds consumed so far.
            max_rounds: Maximum allowed rounds.
            results_so_far: All R-Agent results collected in this loop run.

        Returns:
            A tuple ``(should_stop, reason_string)`` where ``reason_string``
            is a human-readable explanation for paper logging.
        """
        # Derive which strategies have been tried from results_so_far
        tried: set[RefuterStrategy] = {
            r.strategy_used
            for r in results_so_far
            if r.strategy_used is not None
        }

        # Condition 1: all strategies exhausted
        if tried >= set(RefuterStrategy):
            reason = (
                f"all_{len(RefuterStrategy)}_strategies_exhausted_for_conjecture_{conjecture.id}"
            )
            logger.info(
                "Strategist.should_stop | conjecture=%s | stop=True | reason=%s",
                conjecture.id,
                reason,
            )
            return True, reason

        # Condition 2: budget exhausted
        if rounds_used >= max_rounds:
            reason = f"budget_exhausted_rounds_used={rounds_used}_max={max_rounds}"
            logger.info(
                "Strategist.should_stop | conjecture=%s | stop=True | reason=%s",
                conjecture.id,
                reason,
            )
            return True, reason

        # Condition 3: all tried strategies have sub-threshold win rates
        available_untried = set(RefuterStrategy) - tried
        if not available_untried:
            # This path is covered by Condition 1, but be explicit
            return True, "no_strategies_remaining"

        # Check if the best untried strategy has sufficient empirical win rate
        best_win_rate = max(
            self._global_stats[s].win_rate for s in available_untried
        )
        if (
            all(self._global_stats[s].attempts >= _MIN_SAMPLES_FOR_UCB for s in available_untried)
            and best_win_rate < _STOP_CONFIDENCE_THRESHOLD
        ):
            reason = (
                f"low_confidence_best_win_rate={best_win_rate:.4f}_threshold="
                f"{_STOP_CONFIDENCE_THRESHOLD}"
            )
            logger.info(
                "Strategist.should_stop | conjecture=%s | stop=True | reason=%s",
                conjecture.id,
                reason,
            )
            return True, reason

        logger.debug(
            "Strategist.should_stop | conjecture=%s | stop=False | "
            "rounds_used=%d | max_rounds=%d | tried_strategies=%d",
            conjecture.id,
            rounds_used,
            max_rounds,
            len(tried),
        )
        return False, ""

    async def analyze_failure_patterns(
        self,
        results: list[RefuterResult],
    ) -> str:
        """Generate an LLM analysis of why refutation strategies failed.

        Summarises all attempted strategies and their outcomes and asks the
        LLM to identify structural reasons for failure.  The output is intended
        for the research notebook section of the ICML paper.

        Args:
            results: All RefuterResult objects collected during the REFUTE loop.

        Returns:
            A free-text LLM analysis string.  Returns a fallback string if the
            LLM call fails.
        """
        if not results:
            return "No results to analyse — the refutation loop produced no R-Agent outputs."

        strategy_summaries: list[str] = []
        for result in results:
            strategy_name = (
                result.strategy_used.value if result.strategy_used is not None else "unknown"
            )
            n_candidates = len(result.candidates)
            best_ce = (
                result.best_counterexample.candidate_str
                if result.best_counterexample is not None
                else "none"
            )
            strategy_summaries.append(
                f"  - Strategy: {strategy_name} | Candidates: {n_candidates} | "
                f"Best counterexample: {best_ce}"
            )

        conjecture_nl = results[0].conjecture.nl_statement if results else "unknown"
        summary_text = "\n".join(strategy_summaries)

        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are an expert mathematical research analyst. "
                    "Your task is to identify structural reasons why automated "
                    "counterexample search failed for a given conjecture, and "
                    "suggest directions for human follow-up."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The following conjecture survived all automated refutation attempts:\n\n"
                    f"  Conjecture: {conjecture_nl}\n\n"
                    f"Strategy outcomes:\n{summary_text}\n\n"
                    "Please analyse:\n"
                    "1. Why each strategy may have failed structurally.\n"
                    "2. Whether the conjecture is likely true, false but hard, "
                    "or ill-posed.\n"
                    "3. What mathematical techniques might succeed where "
                    "automated search failed.\n\n"
                    "Be concise and precise. Focus on actionable insights for "
                    "a research paper."
                ),
            },
        ]

        try:
            analysis = await self.client.complete(
                messages=messages,
                temperature=_LLM_ANALYSIS_TEMPERATURE,
                max_tokens=_MAX_TOKENS_ANALYSIS,
            )
            logger.debug(
                "Strategist.analyze_failure_patterns | conjecture=%s | analysis_len=%d",
                results[0].conjecture.id,
                len(analysis),
            )
            return analysis
        except RuntimeError as exc:
            logger.warning(
                "Strategist.analyze_failure_patterns | LLM call failed: %s — "
                "returning fallback analysis.",
                exc,
            )
            return (
                f"LLM analysis unavailable (error: {exc}). "
                f"Tried {len(results)} strategies without finding a counterexample."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ucb1_select(
        self,
        available: list[RefuterStrategy],
        domain: Domain,
    ) -> RefuterStrategy:
        """Select a strategy using the UCB1 bandit formula.

        UCB1 score for strategy s:
            score(s) = win_rate(s) + C * sqrt(ln(N) / attempts(s))
        where N = total attempts across all strategies and C is the exploration
        constant.  Domain-level win rates contribute an additive bonus to
        bias selection toward domain-relevant strategies.

        Args:
            available: Strategies that have not yet been tried for this conjecture.
            domain: The domain of the current conjecture.

        Returns:
            The strategy with the highest UCB1 score.
        """
        total_attempts = sum(self._global_stats[s].attempts for s in RefuterStrategy)
        # Avoid log(0) when no data exists
        log_total = math.log(max(total_attempts, 1))

        best_strategy = available[0]
        best_score = -math.inf

        for strategy in available:
            stats = self._global_stats[strategy]
            exploitation = stats.win_rate

            # Exploration bonus — higher for less-tried strategies
            if stats.attempts == 0:
                exploration = float("inf")
            else:
                exploration = _UCB1_EXPLORATION_CONSTANT * math.sqrt(
                    log_total / stats.attempts
                )

            # Domain bonus: add per-domain win rate to encourage domain-matched selection
            domain_bonus = stats.domain_win_rate(domain) * 0.2

            score = exploitation + exploration + domain_bonus

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def _has_domain_analogical_success(self, domain: Domain) -> bool:
        """Check if ANALOGICAL strategy has been successful in the given domain.

        A domain is considered to have analogical success when the per-domain
        win rate for ANALOGICAL is above 20 % and there have been at least 3
        domain-specific attempts.

        Args:
            domain: The mathematical domain to check.

        Returns:
            True if analogical transfer is well-supported for this domain.
        """
        stats = self._global_stats[RefuterStrategy.ANALOGICAL]
        d_key = domain.value
        d_attempts = stats.domain_attempts.get(d_key, 0)
        if d_attempts < 3:
            return False
        d_win_rate = stats.domain_win_rate(domain)
        return d_win_rate >= 0.2

    async def _llm_select_strategy(
        self,
        conjecture: Conjecture,
        available: list[RefuterStrategy],
        past_results: list[RefuterResult],
        global_strategy_stats: dict[str, dict],
    ) -> RefuterStrategy:
        """Use the LLM to recommend a strategy when statistics are sparse.

        Builds a structured prompt describing the conjecture, available
        strategies, their limited statistics, and past round outcomes, then
        parses the LLM's strategy recommendation.  Falls back to the first
        untried strategy in priority order if parsing fails.

        Args:
            conjecture: The conjecture being refuted.
            available: Strategies not yet tried for this conjecture.
            past_results: Completed R-Agent results for earlier rounds.
            global_strategy_stats: External cross-conjecture stats dict
                (may differ from internal _global_stats; used for prompt context).

        Returns:
            The recommended RefuterStrategy.
        """
        strategy_descriptions = {
            RefuterStrategy.BOUNDARY: (
                "Tests boundary / edge cases of the conjecture's domain "
                "(e.g., n=0, 1, 2, extreme values). Very cheap, no LLM needed."
            ),
            RefuterStrategy.RANDOM_STRUCTURED: (
                "Generates random structured candidates respecting the "
                "conjecture's variable constraints. Moderate cost."
            ),
            RefuterStrategy.SYMBOLIC_PERTURBATION: (
                "Symbolically perturbs known counterexamples or near-misses "
                "to find confirming instances. Moderate-high cost."
            ),
            RefuterStrategy.ANALOGICAL: (
                "Transfers counterexample patterns from analogous conjectures "
                "in the same domain. High cost, best for structured domains."
            ),
        }

        past_outcomes: list[str] = []
        for rr in past_results:
            strat = rr.strategy_used.value if rr.strategy_used is not None else "unknown"
            outcome = "found" if rr.best_counterexample is not None else "not_found"
            past_outcomes.append(f"  - {strat}: counterexample={outcome}")

        past_text = (
            "\n".join(past_outcomes) if past_outcomes else "  (no prior rounds)"
        )

        available_text = "\n".join(
            f"  - {s.value}: {strategy_descriptions[s]}" for s in available
        )

        stats_text = json.dumps(
            {k: v for k, v in global_strategy_stats.items()}, indent=2
        )

        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a mathematical research strategy advisor. "
                    "Given a conjecture and past refutation attempts, select "
                    "the single best next strategy. "
                    "Reply with ONLY the strategy name — one of: "
                    + ", ".join(s.value for s in available)
                    + ". No other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conjecture: {conjecture.nl_statement}\n"
                    f"Domain: {conjecture.domain.value}\n\n"
                    f"Past rounds:\n{past_text}\n\n"
                    f"Available strategies:\n{available_text}\n\n"
                    f"Global strategy stats (sparse):\n{stats_text}\n\n"
                    "Which strategy should the R-Agent try next?"
                ),
            },
        ]

        default_strategy = available[0]

        try:
            reply = await self.client.complete(
                messages=messages,
                temperature=_LLM_STRATEGY_TEMPERATURE,
                max_tokens=_MAX_TOKENS_STRATEGY,
            )
        except RuntimeError as exc:
            logger.warning(
                "Strategist._llm_select_strategy | LLM call failed: %s — "
                "falling back to %s",
                exc,
                default_strategy.value,
            )
            return default_strategy

        reply_clean = reply.strip().lower().replace("-", "_")
        for strategy in available:
            if strategy.value.lower() in reply_clean:
                logger.debug(
                    "Strategist._llm_select_strategy | LLM recommended=%s",
                    strategy.value,
                )
                return strategy

        logger.warning(
            "Strategist._llm_select_strategy | Could not parse LLM reply %r — "
            "falling back to %s",
            reply,
            default_strategy.value,
        )
        return default_strategy

    def _log_decision(
        self,
        conjecture_id: str,
        chosen: RefuterStrategy,
        reason: str,
    ) -> None:
        """Append a strategy decision record to the internal decision log.

        Args:
            conjecture_id: ID of the conjecture for which the decision was made.
            chosen: The selected strategy.
            reason: Human-readable reason string for research logging.
        """
        self._decision_log.append(
            {
                "conjecture_id": conjecture_id,
                "chosen_strategy": chosen.value,
                "reason": reason,
                "global_stats_snapshot": {
                    s.value: {
                        "attempts": self._global_stats[s].attempts,
                        "successes": self._global_stats[s].successes,
                        "win_rate": round(self._global_stats[s].win_rate, 4),
                    }
                    for s in RefuterStrategy
                },
            }
        )

    # ------------------------------------------------------------------
    # Inspection helpers (for tests and research notebook)
    # ------------------------------------------------------------------

    def get_stats_summary(self) -> dict[str, dict]:
        """Return a JSON-serialisable summary of all strategy statistics.

        Returns:
            Dict mapping strategy name to a dict with ``attempts``,
            ``successes``, and ``win_rate`` keys.
        """
        return {
            s.value: {
                "attempts": self._global_stats[s].attempts,
                "successes": self._global_stats[s].successes,
                "win_rate": round(self._global_stats[s].win_rate, 4),
                "domain_breakdown": {
                    domain_key: {
                        "attempts": self._global_stats[s].domain_attempts.get(domain_key, 0),
                        "successes": self._global_stats[s].domain_successes.get(domain_key, 0),
                    }
                    for domain_key in self._global_stats[s].domain_attempts
                },
            }
            for s in RefuterStrategy
        }

    def get_decision_log(self) -> list[dict]:
        """Return the full ordered decision log for research auditing.

        Returns:
            A list of decision record dicts in chronological order.
        """
        return list(self._decision_log)
