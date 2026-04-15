"""
Conjecture generation module for the ConjLean pipeline.

Uses an LLMClient to generate batches of mathematical conjectures across
supported domains (number theory, inequality, combinatorics). Handles prompt
construction, batched async LLM calls, output parsing, and deduplication.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm.asyncio import tqdm as async_tqdm

from conjlean.schemas import Conjecture, Domain

if TYPE_CHECKING:
    from conjlean.config import ConjLeanConfig
    from conjlean.models import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific guidance strings injected into user prompts
# ---------------------------------------------------------------------------

_DOMAIN_GUIDANCE: dict[Domain, str] = {
    Domain.NUMBER_THEORY: (
        "Focus on the following sub-areas:\n"
        "  - Divisibility: claims of the form 'k divides f(n) for all n' "
        "(e.g., factorial divisibility, product-of-consecutive-integers arguments)\n"
        "  - GCD properties: identities and bounds involving gcd(f(n), g(n))\n"
        "  - Modular arithmetic: patterns in p^k mod N, quadratic residues, "
        "sum-of-digits mod k\n"
        "  - Primality: properties of primes, prime gaps, prime congruence classes\n"
        "  - Special sequences: Fibonacci, triangular numbers, perfect squares, "
        "sum of divisors\n"
        "Use variables: n, m, k for natural numbers (Nat); p, q for primes.\n"
        "Target difficulty: provable by omega / norm_num / decide, or a short "
        "inductive proof that an LLM can reconstruct."
    ),
    Domain.INEQUALITY: (
        "Focus on the following sub-areas:\n"
        "  - AM-GM variants and generalizations over 2–4 positive reals\n"
        "  - Sum-of-squares decompositions and polynomial non-negativity\n"
        "  - Cauchy-Schwarz instances for finite sums\n"
        "  - Power-mean inequalities (harmonic, geometric, arithmetic)\n"
        "  - Parametrized families: inequalities indexed by a natural number n\n"
        "Use variables: a, b, c for positive reals; n for natural numbers in "
        "parametrized families.\n"
        "All inequalities must be stated over explicit domains "
        "(e.g., 'for all positive reals a, b').\n"
        "Target difficulty: provable by nlinarith / positivity / linarith, "
        "possibly after a ring normalization step."
    ),
    Domain.COMBINATORICS: (
        "CRITICAL CONSTRAINT: generate ONLY bounded or finite claims where "
        "Lean's 'decide' tactic can verify the statement directly.\n"
        "Focus on the following sub-areas:\n"
        "  - Binomial coefficient identities: C(n, k) relationships for n < 20\n"
        "  - Parity arguments over Finsets: sum/product parity for small bounds\n"
        "  - Fibonacci and combinatorial number identities over bounded ranges\n"
        "  - Counting arguments provable by finite enumeration\n"
        "Use variables: n with an EXPLICIT upper bound (n < 20 or n ≤ 15).\n"
        "Every conjecture MUST include an explicit finite bound to be decidable.\n"
        "DO NOT generate open-ended combinatorial identities without a bound."
    ),
}

# ---------------------------------------------------------------------------
# Few-shot examples per domain injected into user prompts
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: dict[Domain, str] = {
    Domain.NUMBER_THEORY: (
        '{"statement": "For all natural numbers n, 3 divides n^3 - n", '
        '"variables": ["n"], "difficulty": "easy"}\n'
        '{"statement": "For all primes p > 2, p^2 - 1 is divisible by 8", '
        '"variables": ["p"], "difficulty": "easy"}\n'
        '{"statement": "For all n >= 1, gcd(n! + 1, (n+1)! + 1) = 1", '
        '"variables": ["n"], "difficulty": "medium"}\n'
        '{"statement": "For all natural numbers n, 6 divides n*(n+1)*(n+2)", '
        '"variables": ["n"], "difficulty": "easy"}\n'
        '{"statement": "For all natural numbers n >= 1, '
        "gcd(2*n + 1, 2*n^2 + 2*n - 1) = 1\", "
        '"variables": ["n"], "difficulty": "medium"}'
    ),
    Domain.INEQUALITY: (
        '{"statement": "For all positive reals a, b: '
        "(a^2 + b^2) / 2 >= ((a + b) / 2)^2\", "
        '"variables": ["a", "b"], "difficulty": "easy"}\n'
        '{"statement": "For all positive reals a, b, c with a + b + c = 1: '
        "a^2 + b^2 + c^2 >= 1/3\", "
        '"variables": ["a", "b", "c"], "difficulty": "medium"}\n'
        '{"statement": "For all positive reals a, b: '
        "(a + b) * (1/a + 1/b) >= 4\", "
        '"variables": ["a", "b"], "difficulty": "easy"}\n'
        '{"statement": "For all positive reals a, b, c: '
        "a^2*b + b^2*c + c^2*a >= a*b*c*(a + b + c) / (a + b + c) * 3\", "
        '"variables": ["a", "b", "c"], "difficulty": "hard"}'
    ),
    Domain.COMBINATORICS: (
        '{"statement": "For all natural numbers n with n < 10, '
        "C(2*n, n) is even\", "
        '"variables": ["n"], "difficulty": "easy"}\n'
        '{"statement": "For all natural numbers n with n < 15, '
        "the sum of the first n odd numbers equals n^2\", "
        '"variables": ["n"], "difficulty": "easy"}\n'
        '{"statement": "For all natural numbers n with n < 12, '
        "C(n, 0) + C(n, 1) + C(n, 2) + ... + C(n, n) = 2^n\", "
        '"variables": ["n"], "difficulty": "medium"}'
    ),
}

# ---------------------------------------------------------------------------
# Prompt file paths (resolved relative to this module)
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt file from the prompts directory.

    Args:
        filename: Filename within the prompts directory.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            "Ensure prompts/ directory is populated before running generation."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ConjectureGenerator
# ---------------------------------------------------------------------------


class ConjectureGenerator:
    """Generates mathematical conjectures using a batched async LLM client.

    Wraps prompt construction, batched LLM calls, output parsing, and
    deduplication into a clean interface. Domain-specific guidance and
    few-shot examples are injected per request to steer the LLM toward
    novel, formalizable, tractable conjectures.

    Attributes:
        client: The LLMClient used for inference.
        config: Pipeline configuration (temperature, batch_size, model name).
        _system_prompt: Loaded system prompt text.
        _user_template: Loaded user prompt template text.
    """

    def __init__(self, client: "LLMClient", config: "ConjLeanConfig") -> None:
        """Initialize the generator with a client and configuration.

        Args:
            client: An LLMClient instance (Anthropic, OpenAI, vLLM, etc.).
            config: ConjLeanConfig with generation and model settings.
        """
        self.client = client
        self.config = config
        self._system_prompt: str = _load_prompt("conjecture_gen_system.txt")
        self._user_template: str = _load_prompt("conjecture_gen_user.txt")
        logger.info(
            "ConjectureGenerator initialized | model=%s | temperature=%.2f | batch_size=%d",
            config.models.conjecture_gen,
            config.generation.temperature,
            config.generation.batch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        domain: Domain,
        n: int,
        existing_ids: set[str] | None = None,
    ) -> list[Conjecture]:
        """Generate n conjectures for a given mathematical domain.

        Issues ceil(n / batch_size) batched LLM calls concurrently, parses
        all outputs, deduplicates by ID, and trims to exactly n results (or
        as many as could be parsed if the LLM under-generates).

        Args:
            domain: The mathematical domain to generate conjectures for.
            n: Target number of conjectures to generate.
            existing_ids: Set of conjecture IDs already in the pipeline;
                conjectures whose generated ID collides with an existing one
                are silently dropped.

        Returns:
            A deduplicated list of Conjecture objects, up to n items.

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        seen_ids: set[str] = set(existing_ids or [])
        batch_size: int = self.config.generation.batch_size
        n_batches: int = (n + batch_size - 1) // batch_size

        # Each batch requests `batch_size` conjectures; last batch may request fewer
        batch_requests: list[int] = []
        remaining = n
        for _ in range(n_batches):
            request_count = min(remaining, batch_size)
            batch_requests.append(request_count)
            remaining -= request_count

        logger.info(
            "Generating %d conjectures for domain=%s across %d batch(es)",
            n,
            domain.value,
            n_batches,
        )

        messages_list: list[list[dict]] = [
            self._build_messages(domain, count) for count in batch_requests
        ]

        raw_outputs: list[str] = await self.client.complete_batch(
            messages_list=messages_list,
            temperature=self.config.generation.temperature,
            max_tokens=4096,
        )

        conjectures: list[Conjecture] = []
        for raw in raw_outputs:
            parsed = self._parse_llm_output(raw, domain)
            for c in parsed:
                if c.id not in seen_ids:
                    seen_ids.add(c.id)
                    conjectures.append(c)

        if len(conjectures) < n:
            logger.warning(
                "Under-generation for domain=%s: requested %d, received %d after dedup",
                domain.value,
                n,
                len(conjectures),
            )
        else:
            conjectures = conjectures[:n]

        logger.info(
            "Generation complete for domain=%s: %d conjectures produced",
            domain.value,
            len(conjectures),
        )
        return conjectures

    async def generate_all_domains(
        self,
        domains: list[Domain],
        n_per_domain: int,
    ) -> list[Conjecture]:
        """Generate conjectures across all specified domains concurrently.

        Launches one async `generate` task per domain and collects results.
        A shared `existing_ids` set is NOT enforced across domains since
        cross-domain ID collisions are extremely unlikely (IDs are prefixed
        with domain name).

        Args:
            domains: List of Domain values to generate for.
            n_per_domain: Number of conjectures to request per domain.

        Returns:
            Flat list of all generated Conjecture objects across all domains.

        Raises:
            ValueError: If domains is empty or n_per_domain < 1.
        """
        if not domains:
            raise ValueError("domains list must not be empty")
        if n_per_domain < 1:
            raise ValueError(f"n_per_domain must be >= 1, got {n_per_domain}")

        logger.info(
            "Launching concurrent generation for %d domain(s), %d each",
            len(domains),
            n_per_domain,
        )

        tasks = [
            asyncio.create_task(
                self.generate(domain=domain, n=n_per_domain),
                name=f"gen_{domain.value}",
            )
            for domain in domains
        ]

        results: list[list[Conjecture]] = await async_tqdm.gather(
            *tasks,
            desc="Domains",
            unit="domain",
            dynamic_ncols=True,
        )

        all_conjectures: list[Conjecture] = [c for batch in results for c in batch]
        logger.info(
            "All-domain generation complete: %d total conjectures across %d domain(s)",
            len(all_conjectures),
            len(domains),
        )
        return all_conjectures

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(self, domain: Domain, n: int) -> list[dict]:
        """Construct the messages list for a single LLM batch call.

        Args:
            domain: The target mathematical domain.
            n: Number of conjectures to request in this call.

        Returns:
            A list of message dicts in OpenAI/Anthropic chat format.
        """
        user_content = self._user_template.format(
            domain=domain.value,
            n=n,
            domain_guidance=_DOMAIN_GUIDANCE[domain],
            few_shot_examples=_FEW_SHOT_EXAMPLES[domain],
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_llm_output(self, raw: str, domain: Domain) -> list[Conjecture]:
        """Parse raw LLM output into a list of Conjecture objects.

        Attempts strict JSON-per-line parsing first. Falls back to heuristic
        line extraction for malformed outputs. Lines that fail all parsing
        strategies are logged and skipped.

        Expected LLM format (one JSON object per line):
            {"statement": "...", "variables": ["n", "m"], "difficulty": "easy|medium|hard"}

        Args:
            raw: The raw text returned by the LLM.
            domain: The domain tag to attach to parsed conjectures.

        Returns:
            List of Conjecture objects successfully parsed from the output.
        """
        conjectures: list[Conjecture] = []
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            # Strip common markdown artifacts (```json ... ```)
            if line.startswith("```"):
                continue

            parsed_obj: dict | None = None

            # --- Strategy 1: strict JSON parse ---
            try:
                parsed_obj = json.loads(line)
            except json.JSONDecodeError:
                pass

            # --- Strategy 2: extract {...} substring ---
            if parsed_obj is None:
                brace_start = line.find("{")
                brace_end = line.rfind("}")
                if brace_start != -1 and brace_end > brace_start:
                    try:
                        parsed_obj = json.loads(line[brace_start : brace_end + 1])
                    except json.JSONDecodeError:
                        pass

            # --- Strategy 3: treat entire line as a plain-text statement ---
            if parsed_obj is None:
                if len(line) > 10:
                    logger.debug(
                        "Line %d: JSON parse failed, using raw line as statement: %.80s",
                        lineno,
                        line,
                    )
                    parsed_obj = {
                        "statement": line,
                        "variables": [],
                        "difficulty": "medium",
                    }
                else:
                    logger.debug("Line %d: skipping short unparseable line: %r", lineno, line)
                    continue

            # --- Validate required fields ---
            statement: str = parsed_obj.get("statement", "").strip()
            if not statement:
                logger.debug("Line %d: empty statement field, skipping", lineno)
                continue

            variables: list[str] = parsed_obj.get("variables", [])
            if not isinstance(variables, list):
                variables = []
            variables = [str(v) for v in variables]

            difficulty: str = parsed_obj.get("difficulty", "medium")
            if difficulty not in {"easy", "medium", "hard"}:
                difficulty = "medium"

            cid = self._generate_id(domain, statement)
            conjecture = Conjecture(
                id=cid,
                domain=domain,
                nl_statement=statement,
                variables=variables,
                source="generated",
                timestamp=timestamp,
                metadata={
                    "difficulty": difficulty,
                    "model": self.config.models.conjecture_gen,
                },
            )
            conjectures.append(conjecture)

        logger.debug(
            "Parsed %d conjectures from LLM output (%d raw lines)",
            len(conjectures),
            len(raw.splitlines()),
        )
        return conjectures

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_id(self, domain: Domain, statement: str) -> str:
        """Generate a deterministic short ID from domain name and statement text.

        Uses SHA-256 of ``"<domain>|<normalized_statement>"`` and takes the
        first 12 hex characters, giving 2^48 ≈ 2.8 × 10^14 possible IDs —
        negligible collision probability at pipeline scale.

        Args:
            domain: The mathematical domain.
            statement: The natural-language conjecture statement.

        Returns:
            A string ID of the form ``"<domain_prefix>_<12-char-hex>"``.
        """
        normalized = " ".join(statement.lower().split())
        payload = f"{domain.value}|{normalized}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        prefix = domain.value[:4]  # e.g., "numb", "ineq", "comb"
        return f"{prefix}_{digest}"
