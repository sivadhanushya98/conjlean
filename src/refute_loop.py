"""
REFUTE Loop: full orchestration of the counterexample-guided conjecture
refinement cycle.

The :class:`RefuteLoop` coordinates the four REFUTE agents in a closed loop:

1. S-Agent (:class:`~conjlean.strategist.Strategist`) selects the next search strategy.
2. R-Agent (``Refuter``) runs the selected strategy to produce candidate counterexamples.
3. V-Agent (external, embedded in the Refuter's verification step) confirms candidates.
4. If a confirmed counterexample is found, C-Agent logic (``_refine_conjecture``)
   proposes a narrowed conjecture and the loop restarts with the refined version.

Terminal conditions:
- ``REFUTED``         — counterexample confirmed, no refinements possible.
- ``REFINED``         — counterexample confirmed and at least one refinement applied.
- ``SURVIVED``        — no counterexample found after all rounds.
- ``BUDGET_EXHAUSTED``— loop terminated by the Strategist's budget signal.

Results are saved incrementally to JSONL after each conjecture in batch mode so
that partial runs survive interruptions.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from tqdm.asyncio import tqdm as async_tqdm

from conjlean.schemas import (
    Conjecture,
    CounterexampleCandidate,
    CounterexampleStatus,
    ConjectureRefinement,
    Domain,
    RefuterResult,
    RefuterStrategy,
    RefuteLoopResult,
    RefuteLoopStatus,
)

if TYPE_CHECKING:
    from conjlean.config import ConjLeanConfig
    from conjlean.models import LLMClient
    from conjlean.strategist import Strategist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REFINEMENT_LLM_TEMPERATURE: float = 0.3
_REFINEMENT_MAX_TOKENS: int = 1024
_DEFAULT_MAX_ROUNDS: int = 10
_DEFAULT_MAX_REFINEMENTS: int = 3
_DEFAULT_MAX_CONCURRENT: int = 4
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# JSON serialisation helpers (mirrors pipeline.py pattern)
# ---------------------------------------------------------------------------


def _recursive_enum_to_value(obj: object) -> object:
    """Recursively convert Enum instances to their ``.value`` in nested structures.

    Mirrors the helper in ``pipeline.py`` to keep serialisation consistent
    across the codebase without introducing a shared utility dependency.

    Args:
        obj: Arbitrary nested structure (dict, list, Enum, or scalar).

    Returns:
        Same structure with all Enum instances replaced by their string values.
    """
    if isinstance(obj, dict):
        return {k: _recursive_enum_to_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_enum_to_value(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _result_to_jsonable(result: RefuteLoopResult) -> dict:
    """Convert a RefuteLoopResult dataclass to a JSON-serialisable dict.

    Args:
        result: A completed RefuteLoopResult.

    Returns:
        A pure-Python dict suitable for ``json.dumps``.
    """
    return _recursive_enum_to_value(dataclasses.asdict(result))


def _append_jsonl(record: dict, path: Path) -> None:
    """Append a single JSON record to a JSONL file, creating it if absent.

    Args:
        record: A JSON-serialisable dict.
        path: Destination file path (parent directories created if absent).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# RefuteLoop
# ---------------------------------------------------------------------------


class RefuteLoop:
    """Full REFUTE loop orchestrating S-Agent, R-Agent, V-Agent, and C-Agent.

    Each call to :meth:`run_single` executes the complete counterexample-
    guided conjecture refinement cycle for one conjecture.  :meth:`run_batch`
    parallelises across a list of conjectures with bounded concurrency via an
    ``asyncio.Semaphore``, emitting incremental JSONL results.

    Components are injected via the constructor to support unit testing and
    alternative backend substitution without subclassing.

    Attributes:
        client: Async LLMClient for conjecture refinement prompts.
        refuter: R-Agent Refuter instance (must expose an async ``run`` method
            that accepts a Conjecture and RefuterStrategy and returns a
            RefuterResult).
        strategist: S-Agent Strategist instance.
        config: Validated ConjLeanConfig.
    """

    def __init__(
        self,
        client: "LLMClient",
        refuter: Any,
        strategist: "Strategist",
        config: "ConjLeanConfig",
    ) -> None:
        """Initialise the RefuteLoop with all required agent components.

        Args:
            client: An async LLMClient used for conjecture refinement prompts.
            refuter: An R-Agent Refuter with an async ``run(conjecture,
                strategy) -> RefuterResult`` interface.
            strategist: A Strategist (S-Agent) instance.
            config: A validated ConjLeanConfig instance.
        """
        self.client = client
        self.refuter = refuter
        self.strategist = strategist
        self.config = config

        # Resolve refute-specific config with graceful fallback
        self._default_max_rounds: int = getattr(
            getattr(config, "refute", None), "max_refute_rounds", _DEFAULT_MAX_ROUNDS
        )
        self._default_max_refinements: int = getattr(
            getattr(config, "refute", None), "max_refinements", _DEFAULT_MAX_REFINEMENTS
        )

        logger.info(
            "RefuteLoop initialised | default_max_rounds=%d | default_max_refinements=%d",
            self._default_max_rounds,
            self._default_max_refinements,
        )

    # ------------------------------------------------------------------
    # Primary public interface
    # ------------------------------------------------------------------

    async def run_single(
        self,
        conjecture: Conjecture,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
        max_refinements: int = _DEFAULT_MAX_REFINEMENTS,
    ) -> RefuteLoopResult:
        """Execute the REFUTE loop on a single conjecture.

        Runs the counterexample-guided refinement cycle:
        1. S-Agent selects a strategy.
        2. R-Agent runs search with that strategy.
        3. On confirmed counterexample: optionally refine conjecture and restart.
        4. On stop signal or budget exhaustion: terminate and return results.

        Args:
            conjecture: The conjecture to refute.
            max_rounds: Maximum total R-Agent rounds across all refinements.
            max_refinements: Maximum number of conjecture refinements allowed
                before terminating with REFINED status.

        Returns:
            A RefuteLoopResult encoding the full trajectory of the loop.
        """
        if not conjecture.id:
            raise ValueError("conjecture.id must not be empty")
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        if max_refinements < 0:
            raise ValueError(f"max_refinements must be >= 0, got {max_refinements}")

        loop_start = time.monotonic()
        original_conjecture = conjecture
        active_conjecture = conjecture

        all_results: list[RefuterResult] = []
        all_refinements: list[ConjectureRefinement] = []
        tried_strategies: set[RefuterStrategy] = set()
        confirmed_counterexample: Optional[CounterexampleCandidate] = None
        total_rounds: int = 0
        final_status: RefuteLoopStatus = RefuteLoopStatus.SURVIVED

        logger.info(
            "RefuteLoop.run_single | conjecture=%s | max_rounds=%d | max_refinements=%d",
            conjecture.id,
            max_rounds,
            max_refinements,
        )

        for round_idx in range(max_rounds):
            total_rounds += 1

            # ── 2a. S-Agent selects strategy ─────────────────────────────
            try:
                strategy = await self.strategist.select_strategy(
                    conjecture=active_conjecture,
                    past_results=all_results,
                    tried_strategies=tried_strategies,
                    global_strategy_stats=self.strategist.get_stats_summary(),
                )
            except ValueError as exc:
                # All strategies exhausted for active conjecture
                logger.info(
                    "RefuteLoop.run_single | round=%d | conjecture=%s | "
                    "strategy selection exhausted: %s",
                    round_idx,
                    active_conjecture.id,
                    exc,
                )
                final_status = RefuteLoopStatus.SURVIVED
                break

            tried_strategies.add(strategy)
            logger.info(
                "RefuteLoop.run_single | round=%d | conjecture=%s | strategy=%s",
                round_idx,
                active_conjecture.id,
                strategy.value,
            )

            # ── 2b. R-Agent runs search ───────────────────────────────────
            refuter_result: RefuterResult = await self._run_refuter_safely(
                conjecture=active_conjecture,
                strategy=strategy,
            )
            all_results.append(refuter_result)

            # ── 2d. S-Agent updates stats ─────────────────────────────────
            success = refuter_result.best_counterexample is not None
            self.strategist.update_stats(
                strategy=strategy,
                success=success,
                domain=active_conjecture.domain,
            )

            # ── 2c. Handle confirmed counterexample ───────────────────────
            if (
                refuter_result.best_counterexample is not None
                and refuter_result.best_counterexample.status
                == CounterexampleStatus.CONFIRMED
            ):
                confirmed_counterexample = refuter_result.best_counterexample
                n_refinements_done = len(all_refinements)

                logger.info(
                    "RefuteLoop.run_single | round=%d | conjecture=%s | "
                    "confirmed counterexample found | refinements_done=%d | "
                    "max_refinements=%d",
                    round_idx,
                    active_conjecture.id,
                    n_refinements_done,
                    max_refinements,
                )

                if n_refinements_done < max_refinements:
                    # ── C-Agent refines the conjecture ────────────────────
                    refinement = await self._refine_conjecture(
                        conjecture=active_conjecture,
                        counterexample=confirmed_counterexample,
                    )
                    all_refinements.append(refinement)

                    refined_conjecture = self._build_refined_conjecture(
                        original=active_conjecture,
                        refinement=refinement,
                    )

                    logger.info(
                        "RefuteLoop.run_single | conjecture refined | "
                        "old_id=%s | new_id=%s | refinement_type=%s",
                        active_conjecture.id,
                        refined_conjecture.id,
                        refinement.refinement_type,
                    )

                    # Reset active conjecture and tried strategies for new round
                    active_conjecture = refined_conjecture
                    tried_strategies = set()
                    # Reset confirmed_counterexample — need to find one for the
                    # refined conjecture before we can declare REFUTED
                    confirmed_counterexample = refuter_result.best_counterexample
                    final_status = RefuteLoopStatus.REFINED
                    continue
                else:
                    # Max refinements reached
                    final_status = RefuteLoopStatus.REFINED
                    break

            # ── 2e. S-Agent stop check ────────────────────────────────────
            should_stop, stop_reason = self.strategist.should_stop(
                conjecture=active_conjecture,
                rounds_used=total_rounds,
                max_rounds=max_rounds,
                results_so_far=all_results,
            )
            if should_stop:
                logger.info(
                    "RefuteLoop.run_single | S-Agent stop | conjecture=%s | "
                    "reason=%s | rounds_used=%d",
                    active_conjecture.id,
                    stop_reason,
                    total_rounds,
                )
                # Distinguish exhausted budget from graceful convergence.
                # The Strategist embeds "budget_exhausted" in the reason string
                # when stopping due to compute limits; other reasons (strategies
                # exhausted, low confidence) indicate graceful SURVIVED.
                if total_rounds >= max_rounds or "budget_exhausted" in stop_reason:
                    final_status = RefuteLoopStatus.BUDGET_EXHAUSTED
                else:
                    final_status = RefuteLoopStatus.SURVIVED
                break
        else:
            # for-loop completed without break → budget exhausted
            if confirmed_counterexample is not None:
                final_status = (
                    RefuteLoopStatus.REFINED
                    if all_refinements
                    else RefuteLoopStatus.REFUTED
                )
            else:
                final_status = RefuteLoopStatus.BUDGET_EXHAUSTED

        # ── Determine terminal status ─────────────────────────────────────
        if confirmed_counterexample is not None and not all_refinements:
            final_status = RefuteLoopStatus.REFUTED
        elif confirmed_counterexample is not None and all_refinements:
            final_status = RefuteLoopStatus.REFINED

        elapsed = time.monotonic() - loop_start
        logger.info(
            "RefuteLoop.run_single | DONE | conjecture=%s | status=%s | "
            "total_rounds=%d | refinements=%d | elapsed=%.2fs",
            original_conjecture.id,
            final_status.value,
            total_rounds,
            len(all_refinements),
            elapsed,
        )

        return RefuteLoopResult(
            original_conjecture=original_conjecture,
            status=final_status,
            refuter_results=all_results,
            refinements=all_refinements,
            final_conjecture=active_conjecture,
            total_rounds=total_rounds,
            confirmed_counterexample=confirmed_counterexample,
        )

    async def run_batch(
        self,
        conjectures: list[Conjecture],
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
        max_refinements: int = _DEFAULT_MAX_REFINEMENTS,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        output_path: Optional[Path] = None,
    ) -> list[RefuteLoopResult]:
        """Run the REFUTE loop on a batch of conjectures with bounded concurrency.

        Dispatches up to ``max_concurrent`` loops simultaneously via an
        ``asyncio.Semaphore``.  Results are saved incrementally to
        ``output_path`` (if provided) so partial runs survive interruptions.
        Progress is tracked via a tqdm bar updated in-place.

        Args:
            conjectures: List of conjectures to process.
            max_rounds: Maximum R-Agent rounds per conjecture.
            max_refinements: Maximum conjecture refinements per conjecture.
            max_concurrent: Maximum simultaneous REFUTE loop coroutines.
            output_path: Optional Path to a JSONL file for incremental saving.
                Entries are appended (file is created if absent).

        Returns:
            List of RefuteLoopResult objects in the same order as input.

        Raises:
            ValueError: If conjectures is empty.
        """
        if not conjectures:
            raise ValueError("conjectures list must not be empty")

        logger.info(
            "RefuteLoop.run_batch | n=%d | max_rounds=%d | max_refinements=%d | "
            "max_concurrent=%d",
            len(conjectures),
            max_rounds,
            max_refinements,
            max_concurrent,
        )

        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[Optional[RefuteLoopResult]] = [None] * len(conjectures)

        async def _bounded_run(idx: int, conjecture: Conjecture) -> None:
            async with semaphore:
                result = await self.run_single(
                    conjecture=conjecture,
                    max_rounds=max_rounds,
                    max_refinements=max_refinements,
                )
                results[idx] = result
                if output_path is not None:
                    _append_jsonl(_result_to_jsonable(result), output_path)
                    logger.debug(
                        "RefuteLoop.run_batch | saved conjecture=%s → %s",
                        conjecture.id,
                        output_path,
                    )

        tasks = [
            asyncio.create_task(
                _bounded_run(idx, conj),
                name=f"refute_{conj.id}",
            )
            for idx, conj in enumerate(conjectures)
        ]

        # Counters for live tqdm postfix
        status_counts: dict[str, int] = {s.value: 0 for s in RefuteLoopStatus}

        pbar = async_tqdm(
            total=len(tasks),
            desc="REFUTE batch",
            unit="conjecture",
            dynamic_ncols=True,
        )

        for coro in asyncio.as_completed(tasks):
            await coro
            # Update status counts from completed results
            status_counts = {s.value: 0 for s in RefuteLoopStatus}
            for r in results:
                if r is not None:
                    status_counts[r.status.value] = (
                        status_counts.get(r.status.value, 0) + 1
                    )
            pbar.set_postfix(
                refuted=status_counts.get(RefuteLoopStatus.REFUTED.value, 0),
                refined=status_counts.get(RefuteLoopStatus.REFINED.value, 0),
                survived=status_counts.get(RefuteLoopStatus.SURVIVED.value, 0),
                exhausted=status_counts.get(RefuteLoopStatus.BUDGET_EXHAUSTED.value, 0),
                refresh=False,
            )
            pbar.update(1)

        pbar.close()

        # All tasks complete — results list is fully populated
        final_results: list[RefuteLoopResult] = [r for r in results if r is not None]

        # Log batch summary
        summary = {s.value: 0 for s in RefuteLoopStatus}
        for r in final_results:
            summary[r.status.value] += 1

        logger.info(
            "RefuteLoop.run_batch | DONE | total=%d | refuted=%d | refined=%d | "
            "survived=%d | budget_exhausted=%d",
            len(final_results),
            summary.get(RefuteLoopStatus.REFUTED.value, 0),
            summary.get(RefuteLoopStatus.REFINED.value, 0),
            summary.get(RefuteLoopStatus.SURVIVED.value, 0),
            summary.get(RefuteLoopStatus.BUDGET_EXHAUSTED.value, 0),
        )

        return final_results

    # ------------------------------------------------------------------
    # Conjecture refinement (C-Agent role)
    # ------------------------------------------------------------------

    async def _refine_conjecture(
        self,
        conjecture: Conjecture,
        counterexample: CounterexampleCandidate,
    ) -> ConjectureRefinement:
        """Invoke the C-Agent LLM to produce a refined conjecture statement.

        Sends a structured prompt containing the original conjecture and the
        confirmed counterexample, then parses the LLM's JSON response into a
        ConjectureRefinement.  Falls back to a minimal heuristic refinement
        if the LLM call fails or the JSON response is malformed.

        Args:
            conjecture: The conjecture that was found to be false.
            counterexample: The confirmed counterexample that disproved it.

        Returns:
            A ConjectureRefinement encoding the updated statement, refinement
            type, and explanatory text.
        """
        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a mathematical conjecture refinement agent. "
                    "Your task is to fix a false conjecture by adding necessary "
                    "conditions that exclude the given counterexample, without "
                    "making the conjecture trivially true."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The following conjecture was found to be FALSE:\n\n"
                    f"  Conjecture: {conjecture.nl_statement}\n"
                    f"  Domain: {conjecture.domain.value}\n"
                    f"  Counterexample: {counterexample.candidate_str}\n\n"
                    "Please refine the conjecture by:\n"
                    "1. Adding the necessary condition(s) that would exclude "
                    "this counterexample\n"
                    "2. Narrowing the domain appropriately\n"
                    "3. Strengthening the hypothesis\n\n"
                    "Return a JSON object (no markdown, no explanation outside "
                    "the JSON):\n"
                    '{"refined_statement": "...", '
                    '"refinement_type": "added_condition|narrowed_domain|'
                    'strengthened_hypothesis|other", '
                    '"explanation": "..."}'
                ),
            },
        ]

        model_name = getattr(self.config.models, "conjecture_gen", "unknown_model")

        try:
            raw_reply = await self.client.complete(
                messages=messages,
                temperature=_REFINEMENT_LLM_TEMPERATURE,
                max_tokens=_REFINEMENT_MAX_TOKENS,
            )
            refinement = self._parse_refinement_reply(
                raw=raw_reply,
                original=conjecture,
                counterexample=counterexample,
                model_name=model_name,
            )
        except RuntimeError as exc:
            logger.warning(
                "RefuteLoop._refine_conjecture | LLM call failed for conjecture=%s: %s — "
                "using heuristic fallback refinement",
                conjecture.id,
                exc,
            )
            refinement = self._heuristic_refinement(
                conjecture=conjecture,
                counterexample=counterexample,
                model_name=model_name,
            )

        logger.debug(
            "RefuteLoop._refine_conjecture | conjecture=%s | refinement_type=%s | "
            "refined_statement=%.120s",
            conjecture.id,
            refinement.refinement_type,
            refinement.refined_statement,
        )
        return refinement

    def _build_refined_conjecture(
        self,
        original: Conjecture,
        refinement: ConjectureRefinement,
    ) -> Conjecture:
        """Construct a new Conjecture dataclass from an original and a refinement.

        The new conjecture inherits the original's domain and variables, gets a
        fresh ID derived from the refined statement, and carries provenance
        metadata linking it to the original.

        Args:
            original: The conjecture before refinement.
            refinement: The C-Agent refinement to apply.

        Returns:
            A new Conjecture with the refined natural-language statement.
        """
        import hashlib
        from datetime import datetime, timezone

        normalized = " ".join(refinement.refined_statement.lower().split())
        payload = f"{original.domain.value}|refined|{normalized}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        prefix = original.domain.value[:4]
        new_id = f"{prefix}_ref_{digest}"

        return Conjecture(
            id=new_id,
            domain=original.domain,
            nl_statement=refinement.refined_statement,
            variables=original.variables,
            source="refined",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            metadata={
                **original.metadata,
                "refined_from": original.id,
                "refinement_type": refinement.refinement_type,
                "original_statement": original.nl_statement,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_refuter_safely(
        self,
        conjecture: Conjecture,
        strategy: RefuterStrategy,
    ) -> RefuterResult:
        """Call the R-Agent with exception isolation and a fallback empty result.

        Wraps the Refuter's ``run`` method to ensure the REFUTE loop can
        continue even if the R-Agent raises an unexpected exception.

        Args:
            conjecture: The conjecture to refute.
            strategy: The strategy to use for this round.

        Returns:
            A RefuterResult — either the real result or a safe empty fallback.
        """
        try:
            result: RefuterResult = await self.refuter.run(
                conjecture=conjecture,
                strategy=strategy,
            )
            logger.debug(
                "RefuteLoop._run_refuter_safely | conjecture=%s | strategy=%s | "
                "candidates=%d | confirmed=%s",
                conjecture.id,
                strategy.value,
                len(result.candidates),
                result.best_counterexample is not None,
            )
            return result
        except Exception as exc:  # noqa: BLE001 — refuter boundary; isolate failures
            logger.error(
                "RefuteLoop._run_refuter_safely | R-Agent raised %s for "
                "conjecture=%s strategy=%s: %s — returning empty result",
                type(exc).__name__,
                conjecture.id,
                strategy.value,
                exc,
            )
            return RefuterResult(
                conjecture=conjecture,
                candidates=[],
                best_counterexample=None,
                strategy_used=strategy,
                rounds=1,
                strategy_scores={strategy.value: 0},
            )

    def _parse_refinement_reply(
        self,
        raw: str,
        original: Conjecture,
        counterexample: CounterexampleCandidate,
        model_name: str,
    ) -> ConjectureRefinement:
        """Parse a raw LLM reply into a ConjectureRefinement.

        Attempts strict JSON parse first, then extracts the first JSON block
        from the reply using a regex.  Falls back to heuristic refinement if
        both strategies fail.

        Args:
            raw: Raw LLM reply text.
            original: The original conjecture.
            counterexample: The counterexample that triggered refinement.
            model_name: Model name for provenance tracking.

        Returns:
            A ConjectureRefinement parsed from the LLM reply or heuristically
            generated.
        """
        parsed: Optional[dict] = None

        # Strategy 1: strict JSON parse
        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract first {...} block
        if parsed is None:
            match = _JSON_BLOCK_RE.search(raw)
            if match is not None:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            logger.warning(
                "RefuteLoop._parse_refinement_reply | Could not parse JSON from LLM "
                "reply for conjecture=%s — using heuristic fallback",
                original.id,
            )
            return self._heuristic_refinement(
                conjecture=original,
                counterexample=counterexample,
                model_name=model_name,
            )

        refined_statement: str = parsed.get("refined_statement", "").strip()
        if not refined_statement:
            logger.warning(
                "RefuteLoop._parse_refinement_reply | Empty refined_statement for "
                "conjecture=%s — using heuristic fallback",
                original.id,
            )
            return self._heuristic_refinement(
                conjecture=original,
                counterexample=counterexample,
                model_name=model_name,
            )

        refinement_type: str = parsed.get("refinement_type", "other").strip()
        valid_types = {"added_condition", "narrowed_domain", "strengthened_hypothesis", "other"}
        if refinement_type not in valid_types:
            refinement_type = "other"

        return ConjectureRefinement(
            original=original,
            refined_statement=refined_statement,
            counterexample_that_prompted=counterexample,
            refinement_type=refinement_type,
            model=model_name,
        )

    def _heuristic_refinement(
        self,
        conjecture: Conjecture,
        counterexample: CounterexampleCandidate,
        model_name: str,
    ) -> ConjectureRefinement:
        """Produce a minimal heuristic refinement when the LLM is unavailable.

        Appends a parenthetical exclusion clause to the original statement
        referencing the counterexample.  This is a conservative fallback that
        ensures the pipeline can continue without LLM availability.

        Args:
            conjecture: The original (false) conjecture.
            counterexample: The confirmed counterexample.
            model_name: Model name (used for provenance; typically "heuristic"
                in fallback calls).

        Returns:
            A ConjectureRefinement with a heuristically constructed statement.
        """
        ce_summary = counterexample.candidate_str[:120]
        refined_statement = (
            f"{conjecture.nl_statement} "
            f"(excluding cases such as: {ce_summary})"
        )
        return ConjectureRefinement(
            original=conjecture,
            refined_statement=refined_statement,
            counterexample_that_prompted=counterexample,
            refinement_type="added_condition",
            model=f"{model_name}_heuristic_fallback",
        )
