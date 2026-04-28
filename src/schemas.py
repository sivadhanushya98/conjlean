"""
Shared type definitions for the ConjLean / REFUTE pipeline.

All dataclasses and enums used across conjecture generation, filtering,
formalization, proof search, counterexample generation, conjecture refinement,
benchmark management, and fine-tuning data preparation are centralized here.
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


# ===========================================================================
# REFUTE — Counterexample-guided conjecture refinement types
# ===========================================================================


class RefuterStrategy(str, Enum):
    """Strategy used by the R-Agent to generate counterexample candidates."""

    BOUNDARY = "boundary"
    RANDOM_STRUCTURED = "random_structured"
    ANALOGICAL = "analogical"
    SYMBOLIC_PERTURBATION = "symbolic_perturbation"


class CounterexampleStatus(str, Enum):
    """Verification outcome of a counterexample candidate."""

    CONFIRMED = "confirmed"
    NOT_CONFIRMED = "not_confirmed"
    UNCERTAIN = "uncertain"


class RefuteLoopStatus(str, Enum):
    """Terminal status of a full REFUTE loop run on a single conjecture."""

    REFUTED = "refuted"           # Confirmed counterexample found
    REFINED = "refined"           # Counterexample found + conjecture refined
    SURVIVED = "survived"         # No counterexample found after all rounds
    BUDGET_EXHAUSTED = "budget_exhausted"  # Stopped due to compute limit


class BenchmarkTier(str, Enum):
    """Tier classification of a benchmark conjecture."""

    TIER1_SYNTHETIC = "tier1_synthetic"   # Known theorems with removed conditions
    TIER2_HISTORICAL = "tier2_historical" # Historical conjectures with known status
    TIER3_SUBTLE = "tier3_subtle"         # Subtle / imprecise published statements


class TrainingSampleSource(str, Enum):
    """Source of a training sample for fine-tuning."""

    FRONTIER_GENERATED = "frontier_generated"
    BENCHMARK_VERIFIED = "benchmark_verified"
    AUGMENTED = "augmented"


@dataclass
class CounterexampleCandidate:
    """
    A single candidate counterexample proposed by the R-Agent.

    Attributes:
        conjecture_id: ID of the conjecture this candidate targets.
        candidate_str: Human-readable counterexample description
            (e.g., ``"n=4: 4^2 - 4 = 12, not divisible by 6"``).
        strategy: The R-Agent strategy that generated this candidate.
        status: Verification outcome from the V-Agent.
        evidence: Supporting numerical evidence from verification.
        reasoning: R-Agent reasoning trace that led to this candidate.
    """

    conjecture_id: str
    candidate_str: str
    strategy: RefuterStrategy
    status: CounterexampleStatus = CounterexampleStatus.UNCERTAIN
    evidence: dict = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class RefuterResult:
    """
    Aggregate result of the R-Agent on one conjecture.

    Attributes:
        conjecture: The conjecture that was attempted.
        candidates: All counterexample candidates explored (ordered by try).
        best_counterexample: The confirmed counterexample if found, else None.
        strategy_used: Strategy that produced the confirmed counterexample.
        rounds: Total number of candidate generation rounds consumed.
        strategy_scores: Per-strategy success counts for S-Agent learning.
    """

    conjecture: Conjecture
    candidates: list[CounterexampleCandidate] = field(default_factory=list)
    best_counterexample: Optional[CounterexampleCandidate] = None
    strategy_used: Optional[RefuterStrategy] = None
    rounds: int = 0
    strategy_scores: dict = field(default_factory=dict)


@dataclass
class ConjectureRefinement:
    """
    A C-Agent refinement of a conjecture after a counterexample is found.

    Attributes:
        original: The conjecture before refinement.
        refined_statement: Updated natural-language statement (e.g., conditions added).
        counterexample_that_prompted: The counterexample that triggered refinement.
        refinement_type: Taxonomy of the change (``"added_condition"``,
            ``"narrowed_domain"``, ``"strengthened_hypothesis"``, ``"other"``).
        model: LLM used for refinement.
    """

    original: Conjecture
    refined_statement: str
    counterexample_that_prompted: CounterexampleCandidate
    refinement_type: str = "added_condition"
    model: str = ""


@dataclass
class RefuteLoopResult:
    """
    Full trajectory of one REFUTE loop run on a single conjecture.

    Attributes:
        original_conjecture: Starting conjecture.
        status: Terminal status of the run.
        refuter_results: Per-round R-Agent results.
        refinements: Sequence of C-Agent refinements (empty if not refined).
        final_conjecture: The conjecture after all refinements (same as original
            if never refined).
        total_rounds: Total rounds consumed across all R-Agent calls.
        confirmed_counterexample: The best confirmed counterexample found.
    """

    original_conjecture: Conjecture
    status: RefuteLoopStatus = RefuteLoopStatus.SURVIVED
    refuter_results: list[RefuterResult] = field(default_factory=list)
    refinements: list[ConjectureRefinement] = field(default_factory=list)
    final_conjecture: Optional[Conjecture] = None
    total_rounds: int = 0
    confirmed_counterexample: Optional[CounterexampleCandidate] = None


@dataclass
class BenchmarkEntry:
    """
    A single entry in the REFUTE 3-tier benchmark.

    Attributes:
        id: Unique benchmark entry ID.
        conjecture: The conjecture to refute.
        tier: Tier classification.
        ground_truth_counterexample: Known counterexample string (for Tier 1 & 2).
        ground_truth_status: Whether the conjecture is known to be false/open.
        source: Human-readable source description (e.g., ``"ProofWiki"``).
        notes: Any relevant annotation or context.
    """

    id: str
    conjecture: Conjecture
    tier: BenchmarkTier
    ground_truth_counterexample: Optional[str] = None
    ground_truth_status: str = "false"  # "false", "open", "true_with_caveat"
    source: str = ""
    notes: str = ""


@dataclass
class TrainingSample:
    """
    A single training sample for fine-tuning the R-Agent (DeepSeek-Math-7B).

    Format follows the supervised fine-tuning schema:
    ``input = conjecture + strategy`` → ``output = reasoning + counterexample``

    Attributes:
        conjecture_nl: Natural-language conjecture statement.
        strategy: R-Agent strategy this sample trains.
        reasoning_trace: Step-by-step reasoning leading to the counterexample.
        counterexample: The confirmed counterexample string.
        verification_evidence: SymPy/SageMath verification output.
        domain: Mathematical domain.
        source: How this sample was created.
    """

    conjecture_nl: str
    strategy: RefuterStrategy
    reasoning_trace: str
    counterexample: str
    verification_evidence: dict = field(default_factory=dict)
    domain: str = "number_theory"
    source: TrainingSampleSource = TrainingSampleSource.FRONTIER_GENERATED
