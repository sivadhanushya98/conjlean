"""
Shared type definitions for the ConjLean pipeline.

All dataclasses and enums used across conjecture generation, filtering,
formalization, and proof search stages are centralized here to enforce a
single source of truth and enable clean dependency graphs between modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Domain(str, Enum):
    """Mathematical domains supported for conjecture generation."""

    NUMBER_THEORY = "number_theory"
    INEQUALITY = "inequality"
    COMBINATORICS = "combinatorics"


class FilterStatus(str, Enum):
    """Outcome of the symbolic / numerical filtering stage."""

    SURVIVING = "surviving"
    DISPROVED = "disproved"
    TRIVIAL = "trivial"


class FormalizationStatus(str, Enum):
    """Outcome of the Lean 4 formalization attempt."""

    TYPECHECKS = "typechecks"
    UNFORMALIZABLE = "unformalizable"


class ProofStatus(str, Enum):
    """Outcome of the layered proof-search stage."""

    PROVED = "proved"
    OPEN = "open"


class ProofLayer(str, Enum):
    """Which proof-search layer closed the proof."""

    LAYER0_AUTO = "layer0_auto"
    LAYER1_COMBO = "layer1_combo"
    LAYER2_SEARCH = "layer2_search"
    LAYER3_LLM = "layer3_llm"


class PipelineStatus(str, Enum):
    """Aggregate status of a conjecture after the full pipeline run."""

    PROVED = "proved"
    OPEN = "open"
    DISPROVED = "disproved"
    UNFORMALIZABLE = "unformalizable"
    FILTERED_OUT = "filtered_out"


@dataclass
class Conjecture:
    """
    A single mathematical conjecture produced by the generation stage.

    Attributes:
        id: Unique identifier (e.g., UUID or ``<domain>_<index>``).
        domain: Mathematical domain this conjecture belongs to.
        nl_statement: Natural-language statement of the conjecture.
        variables: Free variables appearing in the statement.
        source: How the conjecture was produced (default ``"generated"``).
        timestamp: ISO-8601 creation timestamp (empty string if not set).
        metadata: Arbitrary key-value pairs for bookkeeping (prompt hash,
            model name, temperature, etc.).
    """

    id: str
    domain: Domain
    nl_statement: str
    variables: list[str]
    source: str = "generated"
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FilterResult:
    """
    Output of the symbolic / numerical filter applied to a conjecture.

    Attributes:
        conjecture: The conjecture that was evaluated.
        status: Whether it survived, was disproved, or was flagged trivial.
        counterexample: Human-readable counterexample string, if disproved.
        numerical_evidence: Dictionary of spot-checked numerical evaluations
            (e.g., ``{"n=2": True, "n=3": True, ...}``).
    """

    conjecture: Conjecture
    status: FilterStatus
    counterexample: Optional[str] = None
    numerical_evidence: dict = field(default_factory=dict)


@dataclass
class LeanCheckResult:
    """
    Raw result returned by the LeanDojo REPL for a single command.

    Attributes:
        success: True if the command produced no error-severity messages.
        messages: List of message dicts from the REPL
            (keys: ``severity``, ``pos``, ``endPos``, ``data``).
        env_id: The environment ID returned by the REPL after execution.
            Use as the ``env`` field for subsequent chained commands.
    """

    success: bool
    messages: list[dict]
    env_id: int = 0


@dataclass
class FormalizedConjecture:
    """
    A conjecture together with its Lean 4 statement.

    Attributes:
        conjecture: The original conjecture object.
        lean_code: Lean 4 source code encoding the theorem statement
            (typically with ``by sorry``).
        status: Whether the statement typechecks or is unformalizable.
        retries: Number of LLM formalization attempts consumed.
        error_history: Ordered list of Lean error strings encountered during
            iterative refinement.
    """

    conjecture: Conjecture
    lean_code: str
    status: FormalizationStatus
    retries: int = 0
    error_history: list[str] = field(default_factory=list)


@dataclass
class ProofAttempt:
    """
    A single tactic attempt made during proof search.

    Attributes:
        tactic: The tactic string sent to the Lean REPL.
        success: Whether the tactic closed the goal.
        error: Lean error message if the attempt failed.
        layer: Which proof-search layer generated this attempt.
    """

    tactic: str
    success: bool
    error: Optional[str] = None
    layer: Optional[ProofLayer] = None


@dataclass
class ProofResult:
    """
    Aggregate result of the layered proof-search stage for one conjecture.

    Attributes:
        formalized: The formalized conjecture that was attempted.
        status: Whether a proof was found or the conjecture remains open.
        proof: The complete verified Lean 4 proof string, if found.
        layer: Which layer successfully closed the proof.
        attempts: Ordered list of all tactic attempts made.
        duration_seconds: Wall-clock time spent on proof search.
    """

    formalized: FormalizedConjecture
    status: ProofStatus
    proof: Optional[str] = None
    layer: Optional[ProofLayer] = None
    attempts: list[ProofAttempt] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class PipelineResult:
    """
    End-to-end result for a single conjecture through the full pipeline.

    Attributes:
        conjecture: The original conjecture.
        filter_result: Outcome of the symbolic/numerical filter stage.
        formalization: Outcome of the Lean 4 formalization stage.
        proof: Outcome of the proof-search stage.
        final_status: Aggregate pipeline status for reporting and metrics.
    """

    conjecture: Conjecture
    filter_result: Optional[FilterResult] = None
    formalization: Optional[FormalizedConjecture] = None
    proof: Optional[ProofResult] = None
    final_status: PipelineStatus = PipelineStatus.FILTERED_OUT
