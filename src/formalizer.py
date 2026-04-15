"""
Autoformalization layer for the ConjLean pipeline.

Translates natural-language mathematical conjectures into valid Lean 4 theorem
statements (with ``sorry`` proofs) using an LLM-driven error-correction feedback
loop backed by the LeanDojo REPL.

Typical usage::

    formalizer = Formalizer(client, harness, config)
    result = await formalizer.formalize(conjecture)
    results = await formalizer.formalize_batch(conjectures)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm as atqdm

from conjlean.schemas import (
    Conjecture,
    FormalizedConjecture,
    FormalizationStatus,
    LeanCheckResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub interfaces — resolved at runtime via dependency injection.
# Real implementations live in src/llm_client.py and src/lean_harness.py.
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Abstract interface for LLM completion.

    Concrete implementations (AnthropicClient, OpenAIClient, vLLMClient, …)
    must satisfy this contract so that ``Formalizer`` remains provider-agnostic.
    """

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Request a single completion from the language model.

        Args:
            messages: OpenAI-style message list
                (``[{"role": "system"|"user"|"assistant", "content": str}, …]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            Raw model response string.

        Raises:
            NotImplementedError: If the concrete subclass has not implemented
                this method.
        """
        raise NotImplementedError


class LeanHarness:
    """
    Abstract interface for invoking the LeanDojo REPL.

    Concrete implementation wraps the ``lean_dojo`` Python SDK and manages
    the long-running Lean process.
    """

    def check_statement(self, theorem_code: str) -> LeanCheckResult:
        """
        Send a Lean 4 source snippet to the REPL and return the result.

        Args:
            theorem_code: Complete Lean 4 source, typically
                ``import Mathlib\\n\\ntheorem … := by\\n  sorry``.

        Returns:
            :class:`~conjlean.schemas.LeanCheckResult` with ``success``,
            ``messages``, and ``env_id``.

        Raises:
            NotImplementedError: If the concrete subclass has not implemented
                this method.
        """
        raise NotImplementedError


@dataclass
class ConjLeanConfig:
    """
    Lightweight config container for pipeline-level parameters.

    In production this is populated by loading ``configs/config.yaml``
    (e.g., via OmegaConf or PyYAML) and passed through the DI graph.

    Attributes:
        models: Nested namespace with at least a ``formalizer`` attribute
            containing the model name / ID string.
        pipeline: Nested namespace with at least a
            ``max_formalization_retries`` integer attribute.
    """

    @dataclass
    class Models:
        """Model selection sub-config."""

        formalizer: str = "claude-sonnet-4-6"

    @dataclass
    class Pipeline:
        """Pipeline control sub-config."""

        max_formalization_retries: int = 5

    models: Models = field(default_factory=Models)
    pipeline: Pipeline = field(default_factory=Pipeline)


# ---------------------------------------------------------------------------
# Error-type taxonomy and targeted repair hints.
# ---------------------------------------------------------------------------

_ERROR_HINTS: dict[str, str] = {
    "unknown_identifier": (
        "Try fully qualifying: Nat.Prime not Prime, Nat.gcd not gcd, "
        "Nat.factorial not factorial."
    ),
    "type_mismatch": (
        "Check: Nat subtraction truncates (use Int if subtraction can go "
        "negative). Add (↑n : Int) coercions where needed."
    ),
    "ambiguous": (
        "Fully qualify all names. Use Nat.Prime p not Prime p for natural "
        "number primality."
    ),
    "missing_instance": (
        "Add typeclass constraints to the theorem signature, "
        "e.g. [CommRing R] [OrderedField F]."
    ),
    "noncomputable": (
        "Add noncomputable keyword before theorem for Real-valued statements."
    ),
    "universe": (
        "Concretize types: use Nat, Int, or Real instead of polymorphic type "
        "variables."
    ),
    "syntax": (
        "Fix the syntax error. Ensure the theorem has the form "
        "`theorem name (vars) : statement := by\\n  sorry`."
    ),
    "other": (
        "Review the Lean 4 error carefully. Ensure all identifiers are "
        "fully qualified and types are consistent."
    ),
}

# Number of consecutive same-error occurrences before giving up.
_MAX_SAME_ERROR_REPEATS: int = 3
# Number of universe errors before giving up.
_MAX_UNIVERSE_ERRORS: int = 2
# Unfixable kernel error substring.
_KERNEL_FAILURE_SUBSTRING: str = "kernel type check failed"


class Formalizer:
    """
    Translates natural language mathematical conjectures into valid Lean 4
    theorem statements with ``sorry`` proofs, using an LLM with an
    error-correction feedback loop.

    The feedback loop proceeds as follows:

    1. Build an initial ``[system, user]`` message list.
    2. Call the LLM to obtain a candidate Lean 4 theorem statement.
    3. Extract the fenced code block from the response.
    4. Send the code to :meth:`LeanHarness.check_statement`.
    5. If the check succeeds, return a
       :class:`~conjlean.schemas.FormalizedConjecture` with status
       :attr:`~conjlean.schemas.FormalizationStatus.TYPECHECKS`.
    6. Otherwise, build a repair prompt embedding the error and error
       classification, append it to the conversation, and retry.
    7. After ``max_retries`` exhausted (or an early-exit heuristic fires),
       return with status
       :attr:`~conjlean.schemas.FormalizationStatus.UNFORMALIZABLE`.

    Args:
        client: Concrete :class:`LLMClient` implementation.
        harness: Concrete :class:`LeanHarness` implementation.
        config: Pipeline configuration; consults
            ``config.models.formalizer`` and
            ``config.pipeline.max_formalization_retries``.
        system_prompt_path: Path to the system prompt text file.
        repair_prompt_path: Path to the repair prompt template file.
    """

    def __init__(
        self,
        client: LLMClient,
        harness: LeanHarness,
        config: ConjLeanConfig,
        system_prompt_path: str = "prompts/formalizer_system.txt",
        repair_prompt_path: str = "prompts/formalizer_repair.txt",
    ) -> None:
        self._client = client
        self._harness = harness
        self._config = config
        self._max_retries: int = config.pipeline.max_formalization_retries
        self._model: str = config.models.formalizer

        system_path = Path(system_prompt_path)
        repair_path = Path(repair_prompt_path)

        if not system_path.exists():
            raise FileNotFoundError(
                f"System prompt not found at: {system_path.resolve()}"
            )
        if not repair_path.exists():
            raise FileNotFoundError(
                f"Repair prompt not found at: {repair_path.resolve()}"
            )

        self._system_prompt: str = system_path.read_text(encoding="utf-8")
        self._repair_template: str = repair_path.read_text(encoding="utf-8")

        logger.debug(
            "Formalizer initialised | model=%s max_retries=%d",
            self._model,
            self._max_retries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def formalize(self, conjecture: Conjecture) -> FormalizedConjecture:
        """
        Formalize a single conjecture with up to ``max_retries`` error-correction
        rounds.

        Process:

        1. Build initial prompt from system + user messages.
        2. Call LLM to get Lean 4 theorem statement.
        3. Extract code block from LLM response.
        4. Send to :meth:`LeanHarness.check_statement`.
        5. If success → return :class:`~conjlean.schemas.FormalizedConjecture`
           with ``status=TYPECHECKS``.
        6. If error → build repair prompt with error message, retry.
        7. After ``max_retries`` → return with ``status=UNFORMALIZABLE``.

        Every retry is logged at DEBUG level (conjecture id, retry number, error).
        The final outcome is logged at INFO level.

        Args:
            conjecture: The conjecture to formalize.

        Returns:
            A :class:`~conjlean.schemas.FormalizedConjecture` with either
            ``TYPECHECKS`` or ``UNFORMALIZABLE`` status.
        """
        error_history: list[str] = []
        messages: list[dict] = self._build_initial_messages(conjecture)
        lean_code: str = ""
        last_response: str = ""

        for attempt in range(self._max_retries + 1):
            logger.debug(
                "Formalization attempt | conjecture_id=%s retry=%d/%d",
                conjecture.id,
                attempt,
                self._max_retries,
            )

            # --- LLM call ---
            try:
                last_response = await self._client.complete(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024,
                )
            except Exception as exc:  # noqa: BLE001 — re-raised immediately
                logger.error(
                    "LLM completion failed | conjecture_id=%s attempt=%d error=%s",
                    conjecture.id,
                    attempt,
                    exc,
                )
                raise

            # --- Extract code ---
            try:
                lean_code = self._extract_lean_code(last_response)
            except ValueError as exc:
                logger.debug(
                    "Code extraction failed | conjecture_id=%s attempt=%d error=%s",
                    conjecture.id,
                    attempt,
                    exc,
                )
                error_str = f"Code extraction error: {exc}"
                error_history.append(error_str)
                if attempt < self._max_retries:
                    messages = self._build_repair_messages(
                        conjecture=conjecture,
                        failed_code="",
                        error_message=error_str,
                        prior_messages=messages,
                        prior_response=last_response,
                    )
                continue

            # --- Lean type-check (off-loaded to thread pool to avoid blocking the event loop) ---
            check: LeanCheckResult = await asyncio.to_thread(
                self._harness.check_statement, lean_code
            )

            if check.success:
                logger.info(
                    "Formalization SUCCESS | conjecture_id=%s attempts=%d",
                    conjecture.id,
                    attempt + 1,
                )
                return FormalizedConjecture(
                    conjecture=conjecture,
                    lean_code=lean_code,
                    status=FormalizationStatus.TYPECHECKS,
                    retries=attempt,
                    error_history=error_history,
                )

            # --- Collect error ---
            error_message = self._extract_error_text(check)
            error_history.append(error_message)

            logger.debug(
                "Lean typecheck failed | conjecture_id=%s retry=%d error=%r",
                conjecture.id,
                attempt,
                error_message[:200],
            )

            # --- Early-exit heuristic ---
            if self._should_give_up(error_history):
                logger.info(
                    "Early exit triggered | conjecture_id=%s after %d attempts",
                    conjecture.id,
                    attempt + 1,
                )
                break

            if attempt < self._max_retries:
                messages = self._build_repair_messages(
                    conjecture=conjecture,
                    failed_code=lean_code,
                    error_message=error_message,
                    prior_messages=messages,
                    prior_response=last_response,
                )

        logger.info(
            "Formalization FAILED (unformalizable) | conjecture_id=%s total_attempts=%d",
            conjecture.id,
            len(error_history),
        )
        return FormalizedConjecture(
            conjecture=conjecture,
            lean_code=lean_code,
            status=FormalizationStatus.UNFORMALIZABLE,
            retries=min(len(error_history), self._max_retries),
            error_history=error_history,
        )

    async def formalize_batch(
        self,
        conjectures: list[Conjecture],
        max_concurrent: int = 5,
    ) -> list[FormalizedConjecture]:
        """
        Formalize a batch of conjectures with bounded concurrency.

        Uses :class:`asyncio.Semaphore` with ``max_concurrent`` permits to
        prevent overwhelming the LeanDojo REPL with simultaneous requests.
        Displays a ``tqdm`` progress bar updated as each task completes.

        Args:
            conjectures: List of conjectures to formalize.
            max_concurrent: Maximum number of concurrent formalization tasks.
                Defaults to 5.

        Returns:
            List of :class:`~conjlean.schemas.FormalizedConjecture` results
            in the same order as the input ``conjectures`` list.
        """
        if not conjectures:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded(conjecture: Conjecture) -> FormalizedConjecture:
            async with semaphore:
                return await self.formalize(conjecture)

        logger.info(
            "Starting batch formalization | total=%d max_concurrent=%d",
            len(conjectures),
            max_concurrent,
        )

        tasks = [_bounded(c) for c in conjectures]
        results: list[FormalizedConjecture] = []

        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Formalizing",
            unit="conj",
            dynamic_ncols=True,
        ):
            result = await coro
            results.append(result)

        # as_completed does not preserve order; restore input order.
        id_to_result: dict[str, FormalizedConjecture] = {
            r.conjecture.id: r for r in results
        }
        ordered: list[FormalizedConjecture] = [
            id_to_result[c.id] for c in conjectures
        ]

        n_ok = sum(
            1
            for r in ordered
            if r.status == FormalizationStatus.TYPECHECKS
        )
        logger.info(
            "Batch formalization complete | total=%d typechecks=%d unformalizable=%d",
            len(ordered),
            n_ok,
            len(ordered) - n_ok,
        )
        return ordered

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _build_initial_messages(self, conjecture: Conjecture) -> list[dict]:
        """
        Build the initial ``[system, user]`` message list for the LLM.

        The user message includes the conjecture's domain, free variables,
        and natural-language statement to give the model full context.

        Args:
            conjecture: The conjecture to formalize.

        Returns:
            A two-element list of OpenAI-style message dicts.
        """
        user_content = (
            f"Domain: {conjecture.domain.value}\n"
            f"Variables: {', '.join(conjecture.variables) if conjecture.variables else 'none'}\n"
            f"Conjecture: {conjecture.nl_statement}\n\n"
            "Produce the Lean 4 theorem statement for this conjecture."
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _build_repair_messages(
        self,
        conjecture: Conjecture,
        failed_code: str,
        error_message: str,
        prior_messages: list[dict],
        prior_response: str,
    ) -> list[dict]:
        """
        Build the repair conversation by appending the failed attempt and the
        Lean error as assistant + user turns, then requesting a corrected version.

        The repair user message is rendered from ``prompts/formalizer_repair.txt``
        with all placeholders filled in.

        Args:
            conjecture: Original conjecture (provides NL statement for context).
            failed_code: The Lean code that failed typecheck.
            error_message: The exact error string returned by the REPL.
            prior_messages: The message list up to (but not including) the
                most recent assistant response.
            prior_response: The raw LLM response from the most recent attempt
                (used as the assistant turn in the extended conversation).

        Returns:
            Extended message list ready for the next LLM completion call.
        """
        error_type = self._classify_error(error_message)
        targeted_hint = _ERROR_HINTS.get(error_type, _ERROR_HINTS["other"])

        # Determine current retry number from error_history length already
        # accumulated on the FormalizedConjecture (passed indirectly through
        # the caller's loop counter); we approximate from prior_messages depth.
        current_retry = max(0, (len(prior_messages) - 2) // 2 + 1)

        repair_user_content = self._repair_template.format(
            conjecture_nl=conjecture.nl_statement,
            failed_code=failed_code,
            lean_error=error_message,
            error_type=error_type,
            targeted_hint=targeted_hint,
            retry_number=current_retry,
            max_retries=self._max_retries,
        )

        extended = list(prior_messages)
        extended.append({"role": "assistant", "content": prior_response})
        extended.append({"role": "user", "content": repair_user_content})
        return extended

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    def _extract_lean_code(self, llm_response: str) -> str:
        """
        Extract Lean 4 source code from the raw LLM response.

        Extraction strategy (tried in order):

        1. Fenced ``\\`\\`\\`lean … \\`\\`\\``` block.
        2. Generic fenced ``\\`\\`\\` … \\`\\`\\``` block.
        3. Any line starting with ``import Mathlib`` continuing to the end
           of the response.
        4. Raw response stripped of leading/trailing whitespace.

        Args:
            llm_response: The raw string returned by the LLM.

        Returns:
            Extracted Lean 4 source code string.

        Raises:
            ValueError: If the response is empty or contains only whitespace.
        """
        stripped = llm_response.strip()
        if not stripped:
            raise ValueError("LLM response is empty; cannot extract Lean code.")

        # Strategy 1: ```lean ... ```
        lean_fence = re.search(
            r"```lean\s*\n(.*?)```",
            stripped,
            re.DOTALL,
        )
        if lean_fence:
            return lean_fence.group(1).strip()

        # Strategy 2: generic ``` ... ```
        generic_fence = re.search(
            r"```\s*\n(.*?)```",
            stripped,
            re.DOTALL,
        )
        if generic_fence:
            return generic_fence.group(1).strip()

        # Strategy 3: bare import Mathlib … to end of string
        import_match = re.search(
            r"(import Mathlib.*)",
            stripped,
            re.DOTALL,
        )
        if import_match:
            return import_match.group(1).strip()

        # Strategy 4: raw response
        return stripped

    # ------------------------------------------------------------------
    # Error utilities
    # ------------------------------------------------------------------

    def _extract_error_text(self, check: LeanCheckResult) -> str:
        """
        Concatenate all error-severity message ``data`` fields into a single
        string for classification and display.

        Args:
            check: A failed :class:`~conjlean.schemas.LeanCheckResult`.

        Returns:
            Newline-joined error string. Falls back to the full messages
            JSON-repr if no explicit error messages are found.
        """
        errors = [
            msg["data"]
            for msg in check.messages
            if msg.get("severity") == "error" and msg.get("data")
        ]
        if errors:
            return "\n".join(errors)
        # Fallback: include all message data so no information is lost.
        all_data = [
            f"[{msg.get('severity', 'unknown')}] {msg.get('data', '')}"
            for msg in check.messages
            if msg.get("data")
        ]
        return "\n".join(all_data) if all_data else "Unknown Lean error."

    def _classify_error(self, error_message: str) -> str:
        """
        Classify a Lean error string into one of eight canonical categories
        for targeted repair prompt construction.

        Classification is performed by substring matching (case-insensitive)
        in priority order, so the most specific patterns are checked first.

        Categories (returned as strings):

        - ``"unknown_identifier"``: unknown identifier / declaration
        - ``"type_mismatch"``: type mismatch
        - ``"ambiguous"``: ambiguous, possible interpretations
        - ``"missing_instance"``: failed to synthesize instance
        - ``"universe"``: universe level mismatch
        - ``"syntax"``: expected token / unexpected end
        - ``"noncomputable"``: noncomputable
        - ``"other"``: anything else

        Args:
            error_message: Raw Lean error string from the REPL.

        Returns:
            One of the eight category strings listed above.
        """
        lowered = error_message.lower()

        patterns: list[tuple[str, str]] = [
            ("unknown identifier", "unknown_identifier"),
            ("unknown declaration", "unknown_identifier"),
            ("type mismatch", "type_mismatch"),
            ("application type mismatch", "type_mismatch"),
            ("failed to synthesize", "missing_instance"),
            ("could not synthesize", "missing_instance"),
            ("ambiguous", "ambiguous"),
            ("possible interpretations", "ambiguous"),
            ("universe level", "universe"),
            ("universe mismatch", "universe"),
            ("expected token", "syntax"),
            ("unexpected end", "syntax"),
            ("expected '", "syntax"),
            ("noncomputable", "noncomputable"),
            ("kernel type check failed", "other"),
        ]

        for substring, category in patterns:
            if substring in lowered:
                return category

        return "other"

    def _should_give_up(self, error_history: list[str]) -> bool:
        """
        Determine whether to abort the retry loop early based on heuristics
        applied to the accumulated error history.

        Heuristics (checked in order):

        1. ``kernel type check failed`` appears in any error (unfixable by
           the LLM; requires a fundamentally different formulation).
        2. Universe-level errors on two or more attempts.
        3. The same classified error type appears three or more times
           consecutively in the most recent history tail.

        Args:
            error_history: Ordered list of Lean error strings accumulated so
                far (earliest first).

        Returns:
            ``True`` if the retry loop should be aborted immediately.
        """
        if not error_history:
            return False

        # Heuristic 1: unfixable kernel errors.
        if any(_KERNEL_FAILURE_SUBSTRING in e.lower() for e in error_history):
            logger.debug("Give-up: kernel type check failed detected.")
            return True

        # Heuristic 2: repeated universe errors.
        universe_count = sum(
            1
            for e in error_history
            if self._classify_error(e) == "universe"
        )
        if universe_count >= _MAX_UNIVERSE_ERRORS:
            logger.debug(
                "Give-up: %d universe errors accumulated.", universe_count
            )
            return True

        # Heuristic 3: same error type N times in a row.
        if len(error_history) >= _MAX_SAME_ERROR_REPEATS:
            tail = error_history[-_MAX_SAME_ERROR_REPEATS:]
            tail_types = [self._classify_error(e) for e in tail]
            if len(set(tail_types)) == 1:
                logger.debug(
                    "Give-up: error type %r repeated %d times in a row.",
                    tail_types[0],
                    _MAX_SAME_ERROR_REPEATS,
                )
                return True

        return False
