"""
ConjLean — Automated Mathematical Conjecture Generation and Lean 4 Formal Verification.

This package provides the foundation layer for the ConjLean research pipeline:
conjecture generation, symbolic filtering, Lean 4 formalization, and layered
proof search for the ICML 2026 paper.
"""

from __future__ import annotations

from conjlean.schemas import (
    Conjecture,
    Domain,
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
)

__all__ = [
    "Conjecture",
    "Domain",
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
]
