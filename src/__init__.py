"""
ConjLean / REFUTE — Automated counterexample discovery and Lean 4 formal verification.

Two systems live in this package:

1. **ConjLean** (Lean proof pipeline): conjecture generation → symbolic filtering →
   Lean 4 formalization → layered proof search.

2. **REFUTE** (Counterexample pipeline): conjecture loading → R-Agent counterexample
   search (4 strategies) → V-Agent verification → C-Agent conjecture refinement →
   S-Agent meta-control. Benchmark construction + LoRA fine-tuning support included.

ICML AI4Research 2026 submission.
"""

from __future__ import annotations

from conjlean.schemas import (
    # Core conjecture types
    Conjecture,
    Domain,
    # Lean pipeline types
    FilterResult,
    FilterStatus,
    FormalizedConjecture,
    FormalizationStatus,
    LeanCheckResult,
    PipelineResult,
    PipelineStatus,
    ProofAttempt,
    ProofLayer,
    ProofResult,
    ProofStatus,
    # REFUTE types
    CounterexampleCandidate,
    CounterexampleStatus,
    RefuterStrategy,
    RefuterResult,
    ConjectureRefinement,
    RefuteLoopResult,
    RefuteLoopStatus,
    BenchmarkEntry,
    BenchmarkTier,
    TrainingSample,
    TrainingSampleSource,
)

__all__ = [
    # Core
    "Conjecture",
    "Domain",
    # Lean pipeline
    "FilterResult",
    "FilterStatus",
    "FormalizedConjecture",
    "FormalizationStatus",
    "LeanCheckResult",
    "PipelineResult",
    "PipelineStatus",
    "ProofAttempt",
    "ProofLayer",
    "ProofResult",
    "ProofStatus",
    # REFUTE
    "CounterexampleCandidate",
    "CounterexampleStatus",
    "RefuterStrategy",
    "RefuterResult",
    "ConjectureRefinement",
    "RefuteLoopResult",
    "RefuteLoopStatus",
    "BenchmarkEntry",
    "BenchmarkTier",
    "TrainingSample",
    "TrainingSampleSource",
]
