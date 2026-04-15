"""
Multi-layer proof search for Lean 4 theorems in the ConjLean pipeline.

The :class:`ProofSearcher` implements a cascading search strategy with four
escalating layers, each consuming more computational budget than the last:

- **Layer 0** — lightweight automated tactics (``omega``, ``norm_num``, etc.)
- **Layer 1** — structured tactic combinations (induction, ``nlinarith``, etc.)
- **Layer 2** — Lean library-search tactics (``exact?``, ``apply?``)
- **Layer 3** — LLM-guided proof with iterative Lean error feedback

The searcher stops as soon as any layer closes the goal and returns the
verified :class:`~conjlean.schemas.ProofResult`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import textwrap
import threading
import time
from pathlib import Path
from typing import Optional

from conjlean.config import ConjLeanConfig
from conjlean.schemas import (
    FormalizedConjecture,
    FormalizationStatus,
    ProofAttempt,
    ProofLayer,
    ProofResult,
    ProofStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol stubs – runtime implementations are injected via __init__
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin async wrapper around a language model API (injected at runtime)."""

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError


class LeanHarness:
    """Interface to the Lean 4 / Mathlib REPL (injected at runtime)."""

    def try_proof(self, statement_code: str, tactic_body: str) -> object:
        raise NotImplementedError

    def verify_full_proof(self, full_lean_code: str) -> object:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tactic catalogues
# ---------------------------------------------------------------------------

_LAYER0_TACTICS: list[str] = [
    "decide",
    "norm_num",
    "omega",
    "ring",
    "simp",
    "aesop",
    "positivity",
    "simp; omega",
    "simp; ring",
    "simp; norm_num",
    "norm_num [Nat.Prime]",
]

_LAYER1_TACTICS: list[str] = [
    # Induction with simp + omega
    "induction n with\n  | zero => simp\n  | succ n ih => simp [ih]; omega",
    # rcases parity split + omega
    "rcases Nat.even_or_odd n with ⟨k, hk⟩ | ⟨k, hk⟩ <;> omega",
    # nlinarith with standard sq_nonneg hints
    "nlinarith [sq_nonneg a, sq_nonneg b, sq_nonneg (a - b)]",
    "nlinarith [sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (a - c)]",
    # linarith variants
    "linarith",
    "linarith [sq_nonneg a]",
    # positivity variants
    "positivity",
    "apply mul_nonneg <;> positivity",
    # mod_cast / push_cast + omega / ring
    "push_cast; omega",
    "push_cast; ring",
    # field_simp + ring for field identities
    "field_simp; ring",
    # Nat.dvd patterns
    "exact dvd_mul_right _ _",
    "exact dvd_mul_left _ _",
    "apply Nat.dvd_add <;> [exact dvd_mul_right _ _, exact dvd_mul_right _ _]",
]

# Regex for parsing "Try this: exact <lemma>" or "Try this: apply <lemma>"
_TRY_THIS_RE: re.Pattern = re.compile(
    r"Try this:\s+(exact\s+\S+(?:\s+\S+)*|apply\s+\S+(?:\s+\S+)*)",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Prompt template paths (resolved relative to the package root)
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PROOF_GEN_PROMPT = _PACKAGE_ROOT / "prompts" / "proof_gen.txt"
_PROOF_REPAIR_PROMPT = _PACKAGE_ROOT / "prompts" / "proof_repair.txt"


# ---------------------------------------------------------------------------
# ProofSearcher
# ---------------------------------------------------------------------------


class ProofSearcher:
    """
    Multi-layer proof search for Lean 4 theorems.

    Executes a cascade of four layers, stopping at the first success.

    Layer 0: Auto-tactics (``omega``, ``norm_num``, ``decide``, ``ring``,
        ``simp``, ``aesop``, ``positivity``)
    Layer 1: Tactic combinations (induction patterns, ``nlinarith`` with hints,
        ``field_simp``, ``Nat.dvd`` patterns)
    Layer 2: Lean search tactics (``exact?``, ``apply?``) — parse
        ``Try this:`` output and verify the suggested lemma
    Layer 3: LLM-guided proof with Lean error feedback loop (up to
        ``layer3_max_rounds`` dialogue rounds)

    Args:
        client: Async LLM client used exclusively by layer 3.
        harness: Lean 4 REPL harness for all tactic verification.
        config: Validated pipeline configuration object.
    """

    def __init__(
        self,
        client: LLMClient,
        harness: LeanHarness,
        config: ConjLeanConfig,
    ) -> None:
        self._client = client
        self._harness = harness
        self._cfg = config.proof_search

        self._proof_gen_template: str = _load_template(_PROOF_GEN_PROMPT)
        self._proof_repair_template: str = _load_template(_PROOF_REPAIR_PROMPT)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(self, formalized: FormalizedConjecture) -> ProofResult:
        """
        Run the full four-layer cascade and return on first success.

        The method is ``async`` because layer 3 makes awaitable LLM calls.
        Layers 0–2 are synchronous and run in the event-loop thread (they are
        fast enough not to warrant a thread-pool delegation).

        Args:
            formalized: A :class:`~conjlean.schemas.FormalizedConjecture`
                whose ``status`` is ``TYPECHECKS``.

        Returns:
            A :class:`~conjlean.schemas.ProofResult` with
            ``status=PROVED`` if any layer succeeded, otherwise
            ``status=OPEN`` carrying all recorded attempts.

        Raises:
            ValueError: If ``formalized.status`` is not ``TYPECHECKS``.
        """
        if formalized.status is not FormalizationStatus.TYPECHECKS:
            raise ValueError(
                f"Cannot run proof search on conjecture with status "
                f"{formalized.status!r}; expected TYPECHECKS."
            )

        wall_start = time.monotonic()

        # --- Layer 0 ---
        result = self._layer0_auto_tactics(formalized)
        if result is not None:
            result.duration_seconds = time.monotonic() - wall_start
            return result

        # --- Layer 1 ---
        result = self._layer1_combo_tactics(formalized)
        if result is not None:
            result.duration_seconds = time.monotonic() - wall_start
            return result

        # --- Layer 2 ---
        result = self._layer2_search_tactics(formalized)
        if result is not None:
            result.duration_seconds = time.monotonic() - wall_start
            return result

        # --- Layer 3 ---
        result = await self._layer3_llm_proof(formalized)
        if result is not None:
            result.duration_seconds = time.monotonic() - wall_start
            return result

        # All layers exhausted – return OPEN with accumulated attempts
        return ProofResult(
            formalized=formalized,
            status=ProofStatus.OPEN,
            duration_seconds=time.monotonic() - wall_start,
        )

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _layer0_auto_tactics(
        self, formalized: FormalizedConjecture
    ) -> Optional[ProofResult]:
        """
        Try each tactic independently with ``layer0_timeout``.

        Iterates through :data:`_LAYER0_TACTICS` in order and returns a
        ``PROVED`` result on the first successful tactic.  All failed
        attempts are accumulated in the returned ``ProofResult`` for
        diagnostics.

        Args:
            formalized: The conjecture to attempt.

        Returns:
            A :class:`~conjlean.schemas.ProofResult` with
            ``status=PROVED`` on first success, or ``None`` if every
            tactic fails.
        """
        attempts: list[ProofAttempt] = []

        for tactic in _LAYER0_TACTICS:
            attempt = self._try_tactic_with_timeout(
                formalized=formalized,
                tactic=tactic,
                timeout=self._cfg.layer0_timeout,
                layer=ProofLayer.LAYER0_AUTO,
            )
            attempts.append(attempt)

            if attempt.success:
                full_proof = _build_full_proof(formalized.lean_code, tactic)
                logger.info(
                    "Layer 0 closed proof for %s with tactic: %r",
                    formalized.conjecture.id,
                    tactic,
                )
                return ProofResult(
                    formalized=formalized,
                    status=ProofStatus.PROVED,
                    proof=full_proof,
                    layer=ProofLayer.LAYER0_AUTO,
                    attempts=attempts,
                )

        logger.debug("Layer 0 exhausted for %s", formalized.conjecture.id)
        return None

    def _layer1_combo_tactics(
        self, formalized: FormalizedConjecture
    ) -> Optional[ProofResult]:
        """
        Try structured tactic combinations with ``layer1_timeout``.

        Covers induction patterns, ``nlinarith`` with algebraic hints,
        ``linarith``, ``positivity``, ``push_cast``, ``field_simp``, and
        ``Nat.dvd`` lemma applications.

        Args:
            formalized: The conjecture to attempt.

        Returns:
            A :class:`~conjlean.schemas.ProofResult` with
            ``status=PROVED`` on first success, or ``None`` if every
            combination fails.
        """
        attempts: list[ProofAttempt] = []

        for tactic in _LAYER1_TACTICS:
            attempt = self._try_tactic_with_timeout(
                formalized=formalized,
                tactic=tactic,
                timeout=self._cfg.layer1_timeout,
                layer=ProofLayer.LAYER1_COMBO,
            )
            attempts.append(attempt)

            if attempt.success:
                full_proof = _build_full_proof(formalized.lean_code, tactic)
                logger.info(
                    "Layer 1 closed proof for %s with tactic: %r",
                    formalized.conjecture.id,
                    tactic[:60],
                )
                return ProofResult(
                    formalized=formalized,
                    status=ProofStatus.PROVED,
                    proof=full_proof,
                    layer=ProofLayer.LAYER1_COMBO,
                    attempts=attempts,
                )

        logger.debug("Layer 1 exhausted for %s", formalized.conjecture.id)
        return None

    def _layer2_search_tactics(
        self, formalized: FormalizedConjecture
    ) -> Optional[ProofResult]:
        """
        Use Lean's ``exact?`` and ``apply?`` with ``layer2_timeout``.

        These tactics search Mathlib for applicable lemmas and emit a
        ``Try this:`` suggestion line.  This method parses that output,
        extracts the suggested tactic, and re-verifies it to confirm
        soundness.

        Args:
            formalized: The conjecture to attempt.

        Returns:
            A :class:`~conjlean.schemas.ProofResult` with
            ``status=PROVED`` on success, or ``None`` if no suggestion was
            found or verification failed.
        """
        attempts: list[ProofAttempt] = []

        for search_tactic in ("exact?", "apply?"):
            attempt, raw_messages = self._try_tactic_with_timeout_messages(
                formalized=formalized,
                tactic=search_tactic,
                timeout=self._cfg.layer2_timeout,
                layer=ProofLayer.LAYER2_SEARCH,
            )
            attempts.append(attempt)

            # ``exact?`` / ``apply?`` emit ``Try this:`` as an ``info``-severity
            # message, which is not captured in attempt.error (errors/warnings only).
            # Search all raw messages instead.
            all_message_text = "\n".join(m.get("data", "") for m in raw_messages)
            suggestion = _parse_try_this(all_message_text)
            if suggestion:
                verify_attempt = self._try_tactic_with_timeout(
                    formalized=formalized,
                    tactic=suggestion,
                    timeout=self._cfg.layer2_timeout,
                    layer=ProofLayer.LAYER2_SEARCH,
                )
                attempts.append(verify_attempt)

                if verify_attempt.success:
                    full_proof = _build_full_proof(formalized.lean_code, suggestion)
                    logger.info(
                        "Layer 2 closed proof for %s via %r suggestion: %r",
                        formalized.conjecture.id,
                        search_tactic,
                        suggestion[:80],
                    )
                    return ProofResult(
                        formalized=formalized,
                        status=ProofStatus.PROVED,
                        proof=full_proof,
                        layer=ProofLayer.LAYER2_SEARCH,
                        attempts=attempts,
                    )

            elif attempt.success:
                full_proof = _build_full_proof(formalized.lean_code, search_tactic)
                logger.info(
                    "Layer 2 closed proof for %s directly with %r",
                    formalized.conjecture.id,
                    search_tactic,
                )
                return ProofResult(
                    formalized=formalized,
                    status=ProofStatus.PROVED,
                    proof=full_proof,
                    layer=ProofLayer.LAYER2_SEARCH,
                    attempts=attempts,
                )

        logger.debug("Layer 2 exhausted for %s", formalized.conjecture.id)
        return None

    async def _layer3_llm_proof(
        self, formalized: FormalizedConjecture
    ) -> Optional[ProofResult]:
        """
        LLM-guided proof with iterative Lean error feedback.

        Each round of the loop:

        1. Builds the appropriate prompt (``proof_gen.txt`` for round 1,
           ``proof_repair.txt`` for subsequent rounds).
        2. Calls the LLM asynchronously.
        3. Extracts the tactic body from the response.
        4. Verifies via :meth:`~LeanHarness.try_proof`.
        5. On success, confirms with :meth:`~LeanHarness.verify_full_proof`.
        6. On failure, carries the error into the next repair round.

        Args:
            formalized: The conjecture to attempt.

        Returns:
            A :class:`~conjlean.schemas.ProofResult` with
            ``status=PROVED`` if the LLM finds a valid proof within
            ``layer3_max_rounds`` rounds, otherwise ``None``.
        """
        attempts: list[ProofAttempt] = []
        domain = formalized.conjecture.domain.value
        statement_code = formalized.lean_code

        # Extract a rough goal hint from the lean_code (everything after := by sorry)
        goal_hint = _extract_goal_hint(statement_code)

        last_error: Optional[str] = None
        last_tactic: Optional[str] = None

        for round_idx in range(1, self._cfg.layer3_max_rounds + 1):
            # --- Build prompt ---
            if round_idx == 1:
                user_content = self._proof_gen_template.format(
                    theorem_code=statement_code,
                    goal_state_hint=goal_hint,
                    domain=domain,
                )
            else:
                user_content = self._proof_repair_template.format(
                    theorem_code=statement_code,
                    failed_tactic=last_tactic or "",
                    lean_error=last_error or "",
                    round=round_idx,
                    max_rounds=self._cfg.layer3_max_rounds,
                )

            messages = [{"role": "user", "content": user_content}]

            # --- LLM call ---
            try:
                raw_response = await asyncio.wait_for(
                    self._client.complete(
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1024,
                    ),
                    timeout=self._cfg.layer3_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Layer 3 LLM call timed out at round %d for %s",
                    round_idx,
                    formalized.conjecture.id,
                )
                break
            except Exception as exc:
                logger.error(
                    "Layer 3 LLM call failed at round %d for %s: %s",
                    round_idx,
                    formalized.conjecture.id,
                    exc,
                )
                break

            # --- Extract tactic body ---
            tactic_body = self._extract_tactic_body(raw_response)
            if not tactic_body:
                logger.warning(
                    "Layer 3 round %d: could not extract tactic from LLM response for %s",
                    round_idx,
                    formalized.conjecture.id,
                )
                last_error = "LLM returned an empty or unrecognised tactic block."
                last_tactic = ""
                attempts.append(
                    ProofAttempt(
                        tactic="<unparseable>",
                        success=False,
                        error=last_error,
                        layer=ProofLayer.LAYER3_LLM,
                    )
                )
                continue

            # --- Verify with Lean ---
            attempt = self._try_tactic_with_timeout(
                formalized=formalized,
                tactic=tactic_body,
                timeout=self._cfg.layer3_timeout,
                layer=ProofLayer.LAYER3_LLM,
            )
            attempts.append(attempt)
            last_tactic = tactic_body
            last_error = attempt.error

            if attempt.success:
                # Double-check with full proof verification
                full_proof = _build_full_proof(statement_code, tactic_body)
                verify_result = self._harness.verify_full_proof(full_proof)

                if verify_result.success:
                    logger.info(
                        "Layer 3 closed proof for %s at round %d",
                        formalized.conjecture.id,
                        round_idx,
                    )
                    return ProofResult(
                        formalized=formalized,
                        status=ProofStatus.PROVED,
                        proof=full_proof,
                        layer=ProofLayer.LAYER3_LLM,
                        attempts=attempts,
                    )
                else:
                    # try_proof succeeded but full verify failed — treat as error
                    last_error = _format_lean_messages(verify_result.messages)
                    logger.debug(
                        "Layer 3 round %d: try_proof ok but verify_full_proof failed for %s",
                        round_idx,
                        formalized.conjecture.id,
                    )
                    attempts[-1] = ProofAttempt(
                        tactic=tactic_body,
                        success=False,
                        error=last_error,
                        layer=ProofLayer.LAYER3_LLM,
                    )
            else:
                logger.debug(
                    "Layer 3 round %d failed for %s: %s",
                    round_idx,
                    formalized.conjecture.id,
                    (last_error or "")[:120],
                )

        logger.debug(
            "Layer 3 exhausted all %d rounds for %s",
            self._cfg.layer3_max_rounds,
            formalized.conjecture.id,
        )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_tactic_body(self, llm_response: str) -> str:
        """
        Extract the tactic body (text after ``:= by``) from an LLM response.

        Handles three common response formats:

        1. Fenced code block (triple backticks, optionally labelled ``lean``)
        2. Full theorem including ``:= by`` followed by the tactic block
        3. Bare tactic block (no surrounding boilerplate)

        The extracted text is stripped of leading/trailing whitespace and
        de-indented so that the leftmost non-blank line has zero indentation.

        Args:
            llm_response: Raw string returned by the LLM.

        Returns:
            The cleaned tactic body string, or an empty string if extraction
            fails.
        """
        if not llm_response:
            return ""

        # --- 1. Fenced code block ---
        fence_pattern = re.compile(
            r"```(?:lean)?\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )
        fence_match = fence_pattern.search(llm_response)
        if fence_match:
            block = fence_match.group(1)
            # If the block contains ":= by", extract only the tactic part
            block = _strip_theorem_wrapper(block)
            return textwrap.dedent(block).strip()

        # --- 2. Full theorem statement with := by ---
        by_match = re.search(r":=\s*by\s*\n?(.*)", llm_response, re.DOTALL)
        if by_match:
            return textwrap.dedent(by_match.group(1)).strip()

        # --- 3. Bare tactic block ---
        # Heuristic: if the response starts with a known tactic keyword or
        # whitespace, treat the whole response as the tactic body.
        tactic_keywords = (
            "intro", "apply", "exact", "simp", "omega", "ring", "norm_num",
            "decide", "aesop", "linarith", "nlinarith", "positivity",
            "induction", "cases", "rcases", "have", "rw", "constructor",
            "refine", "use", "push_cast", "field_simp",
        )
        stripped = llm_response.strip()
        first_word = stripped.split()[0] if stripped.split() else ""
        if first_word.lower() in tactic_keywords:
            return textwrap.dedent(stripped)

        # Last resort: return the whole stripped response
        return stripped

    def _try_tactic_with_timeout(
        self,
        formalized: FormalizedConjecture,
        tactic: str,
        timeout: int,
        layer: ProofLayer,
    ) -> ProofAttempt:
        """
        Try a single tactic against the Lean harness with a wall-clock timeout.

        Args:
            formalized: The conjecture whose statement code is used.
            tactic: The tactic string to evaluate.
            timeout: Maximum seconds to wait.
            layer: The proof layer this attempt belongs to.

        Returns:
            A :class:`~conjlean.schemas.ProofAttempt` recording success,
            error, and layer.
        """
        attempt, _ = self._try_tactic_with_timeout_messages(
            formalized=formalized,
            tactic=tactic,
            timeout=timeout,
            layer=layer,
        )
        return attempt

    def _try_tactic_with_timeout_messages(
        self,
        formalized: FormalizedConjecture,
        tactic: str,
        timeout: int,
        layer: ProofLayer,
    ) -> tuple[ProofAttempt, list[dict]]:
        """
        Try a single tactic and return both the attempt and raw REPL messages.

        Spawns a daemon thread that calls :meth:`~LeanHarness.try_proof` and
        joins with ``timeout`` seconds.  If the thread does not complete in
        time, the attempt is marked as failed with a timeout error message.

        Args:
            formalized: The conjecture whose statement code is used.
            tactic: The tactic string to evaluate.
            timeout: Maximum seconds to wait.
            layer: The proof layer this attempt belongs to.

        Returns:
            Tuple of (:class:`~conjlean.schemas.ProofAttempt`, raw message list).
            The raw message list includes all severity levels (info, warning, error)
            so callers can parse ``Try this:`` suggestions from info messages.
        """
        result_container: dict = {}

        def _run() -> None:
            try:
                lean_result = self._harness.try_proof(formalized.lean_code, tactic)
                result_container["lean_result"] = lean_result
            except Exception as exc:  # noqa: BLE001 – re-surfaced below
                result_container["exception"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        empty_messages: list[dict] = []

        if thread.is_alive():
            return (
                ProofAttempt(
                    tactic=tactic,
                    success=False,
                    error=f"Tactic timed out after {timeout}s",
                    layer=layer,
                ),
                empty_messages,
            )

        if "exception" in result_container:
            exc = result_container["exception"]
            logger.warning("Lean harness raised exception for tactic %r: %s", tactic[:60], exc)
            return (
                ProofAttempt(
                    tactic=tactic,
                    success=False,
                    error=str(exc),
                    layer=layer,
                ),
                empty_messages,
            )

        lean_result = result_container.get("lean_result")
        if lean_result is None:
            return (
                ProofAttempt(
                    tactic=tactic,
                    success=False,
                    error="Harness returned no result",
                    layer=layer,
                ),
                empty_messages,
            )

        error_msg: Optional[str] = None
        if not lean_result.success:
            error_msg = _format_lean_messages(lean_result.messages)

        return (
            ProofAttempt(
                tactic=tactic,
                success=lean_result.success,
                error=error_msg,
                layer=layer,
            ),
            lean_result.messages,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_template(path: Path) -> str:
    """
    Load a prompt template from disk.

    Args:
        path: Absolute path to the template file.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _build_full_proof(lean_code: str, tactic_body: str) -> str:
    """
    Replace the placeholder ``:= by sorry`` in ``lean_code`` with ``tactic_body``.

    If the placeholder is absent, appends ``:= by\\n  <tactic_body>`` to the
    statement verbatim.

    Args:
        lean_code: Lean 4 statement ending with ``:= by sorry``.
        tactic_body: Verified tactic body to substitute.

    Returns:
        Complete, self-contained Lean 4 proof source string.
    """
    indented = textwrap.indent(tactic_body, "  ")
    sorry_pattern = re.compile(r":=\s*by\s+sorry\s*$", re.MULTILINE)
    if sorry_pattern.search(lean_code):
        return sorry_pattern.sub(f":= by\n{indented}", lean_code)
    return lean_code.rstrip() + f"\n  := by\n{indented}"


def _strip_theorem_wrapper(block: str) -> str:
    """
    If ``block`` contains a full theorem declaration, extract only the
    tactic body appearing after ``:= by``.

    Args:
        block: Potentially full Lean source extracted from a code fence.

    Returns:
        Tactic body string, or the original ``block`` unchanged.
    """
    match = re.search(r":=\s*by\s*\n?(.*)", block, re.DOTALL)
    if match:
        return match.group(1)
    return block


def _parse_try_this(message_text: str) -> Optional[str]:
    """
    Parse a ``Try this:`` suggestion from Lean REPL message text.

    Args:
        message_text: Combined string of Lean REPL messages.

    Returns:
        The extracted tactic string (e.g. ``exact Nat.dvd_mul_right n k``),
        or ``None`` if no suggestion is present.
    """
    match = _TRY_THIS_RE.search(message_text)
    if match:
        return match.group(1).strip()
    return None


def _format_lean_messages(messages: list[dict]) -> str:
    """
    Flatten a list of Lean REPL message dicts into a human-readable string.

    Only ``error`` and ``warning`` severity messages are included.

    Args:
        messages: List of message dicts with keys ``severity`` and ``data``.

    Returns:
        Newline-separated string of relevant messages.
    """
    relevant = [
        m.get("data", "")
        for m in messages
        if m.get("severity") in ("error", "warning")
    ]
    return "\n".join(relevant)


def _extract_goal_hint(lean_code: str) -> str:
    """
    Heuristically extract the logical goal from a Lean statement string.

    Looks for the return type annotation appearing after the last ``:`` that
    precedes ``:=``.

    Args:
        lean_code: Lean 4 source of the theorem statement.

    Returns:
        Best-effort goal hint string for prompt construction.
    """
    # Try to find "theorem <name> ... : <goal> :="
    match = re.search(r":\s*(.+?)\s*:=", lean_code, re.DOTALL)
    if match:
        return match.group(1).strip().replace("\n", " ")
    return ""
