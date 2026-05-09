"""
Symbolic and numerical filtering module for the ConjLean pipeline.

Applies conservative falsification logic to mathematical conjectures using
SymPy for symbolic reasoning and numerical sampling. A conjecture is only
classified as DISPROVED when a concrete counterexample is found; all
unparseable or ambiguous cases survive the filter.

Design principle: false negatives (passing a false conjecture) are acceptable;
false positives (incorrectly discarding a true conjecture) are not.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import random
import re
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tqdm import tqdm

from conjlean.schemas import Conjecture, Domain, FilterResult, FilterStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout sentinel for subprocess-based checks
# ---------------------------------------------------------------------------

_SYMPY_TIMEOUT_SECONDS: int = 5


@dataclass
class _CheckTask:
    """Internal task description passed to a worker process."""

    conjecture: Conjecture
    n_test_values: int
    n_random_attempts: int


@dataclass
class _CheckResult:
    """Internal result returned from a worker process."""

    status: FilterStatus
    counterexample: str | None = None
    numerical_evidence: dict = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess with a SIGALRM-based timeout)
# ---------------------------------------------------------------------------


def _worker_check(task: _CheckTask) -> _CheckResult:
    """Top-level function executed in a worker process for one conjecture.

    Uses SIGALRM on POSIX systems to enforce a hard timeout on SymPy
    evaluation. On timeout, the conjecture is conservatively classified
    as SURVIVING.

    Args:
        task: A _CheckTask describing the conjecture and sampling parameters.

    Returns:
        A _CheckResult with the outcome and any evidence collected.
    """

    def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ANN001
        raise TimeoutError("SymPy evaluation exceeded time limit")

    try:
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(_SYMPY_TIMEOUT_SECONDS)
        result = _dispatch_check(task)
        if hasattr(signal, "alarm"):
            signal.alarm(0)
        return result
    except TimeoutError:
        return _CheckResult(
            status=FilterStatus.SURVIVING,
            error="timeout",
        )
    except Exception as exc:  # noqa: BLE001 — subprocess boundary; must catch all
        return _CheckResult(
            status=FilterStatus.SURVIVING,
            error=f"unexpected error: {type(exc).__name__}: {exc}",
        )


def _dispatch_check(task: _CheckTask) -> _CheckResult:
    """Route a check task to the appropriate domain-specific checker.

    Args:
        task: The _CheckTask to evaluate.

    Returns:
        A _CheckResult from the domain-specific checker.
    """
    checker = _SympyCheckers(
        n_test_values=task.n_test_values,
        n_random_attempts=task.n_random_attempts,
    )

    if task.conjecture.domain == Domain.NUMBER_THEORY:
        return checker.check_number_theory(task.conjecture)
    elif task.conjecture.domain == Domain.INEQUALITY:
        return checker.check_inequality(task.conjecture)
    elif task.conjecture.domain == Domain.COMBINATORICS:
        return checker.check_combinatorics(task.conjecture)
    else:
        return _CheckResult(status=FilterStatus.SURVIVING)


# ---------------------------------------------------------------------------
# Internal checker class (instantiated inside worker processes)
# ---------------------------------------------------------------------------


class _SympyCheckers:
    """Domain-specific SymPy-based checkers.

    All methods are designed to be conservative: they only return DISPROVED
    when a concrete, verified counterexample is in hand.

    Attributes:
        n_test_values: Number of systematic test values to try.
        n_random_attempts: Number of random attempts to try.
    """

    def __init__(self, n_test_values: int, n_random_attempts: int) -> None:
        """Initialize checkers with sampling parameters.

        Args:
            n_test_values: Number of systematic test points.
            n_random_attempts: Number of random test points.
        """
        self.n_test_values = n_test_values
        self.n_random_attempts = n_random_attempts

    # ------------------------------------------------------------------
    # Number theory
    # ------------------------------------------------------------------

    def check_number_theory(self, conjecture: Conjecture) -> _CheckResult:
        """Run number-theory-specific falsification checks.

        Attempts to match the statement against known patterns:
        - Divisibility: "k divides f(n)"
        - Modular: "f(n) mod m = c" or "f(n) % m = c"
        - GCD: "gcd(f(n), g(n)) = k"

        If a pattern matches, tests are run over systematic + random integer
        values. Falls through to SURVIVING if no pattern matches.

        Args:
            conjecture: The number theory conjecture to check.

        Returns:
            A _CheckResult with the outcome.
        """
        import sympy  # noqa: PLC0415

        stmt = conjecture.nl_statement.lower()
        evidence: dict[str, bool] = {}

        # --- Divisibility pattern: "k divides f(n)" or "f(n) is divisible by k" ---
        divisibility_result = self._check_divisibility_pattern(conjecture, stmt, evidence)
        if divisibility_result is not None:
            return divisibility_result

        # --- Modular pattern: "f(n) mod m = c" or "f(n) % m = c" ---
        modular_result = self._check_modular_pattern(conjecture, stmt, evidence)
        if modular_result is not None:
            return modular_result

        # --- GCD pattern: "gcd(f(n), g(n)) = k" ---
        gcd_result = self._check_gcd_pattern(conjecture, stmt, evidence)
        if gcd_result is not None:
            return gcd_result

        # --- Trivial check ---
        if self._is_trivial_statement(stmt):
            return _CheckResult(status=FilterStatus.TRIVIAL, numerical_evidence=evidence)

        # No matchable pattern — conservative pass
        return _CheckResult(status=FilterStatus.SURVIVING, numerical_evidence=evidence)

    def _check_divisibility_pattern(
        self,
        conjecture: Conjecture,
        stmt: str,
        evidence: dict,
    ) -> _CheckResult | None:
        """Check divisibility-type statements numerically.

        Handles patterns like:
        - "k divides f(n)" / "k | f(n)"
        - "f(n) is divisible by k"
        - "k divides n^a - n" etc.

        Args:
            conjecture: The conjecture being checked.
            stmt: Lowercase statement text.
            evidence: Mutable dict to append test results into.

        Returns:
            A _CheckResult if a pattern was matched, None otherwise.
        """
        import sympy  # noqa: PLC0415

        # Pattern: "<k> divides <expr>" where k is a small integer
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
            except ValueError:
                continue

            if k <= 0:
                continue

            expr_str = expr_str.strip().rstrip(".")
            n_sym = sympy.Symbol("n", positive=True, integer=True)
            p_sym = sympy.Symbol("p", positive=True, integer=True)

            try:
                expr = sympy.sympify(
                    expr_str,
                    locals={"n": n_sym, "p": p_sym, "factorial": sympy.factorial},
                )
            except (sympy.SympifyError, SyntaxError):
                continue

            test_vals = self._generate_integer_test_values()
            for val in test_vals:
                try:
                    result = int(expr.subs(n_sym, val).evalf())
                    passes = result % k == 0
                    evidence[f"n={val}"] = passes
                    if not passes:
                        return _CheckResult(
                            status=FilterStatus.DISPROVED,
                            counterexample=f"n={val}: {result} mod {k} = {result % k} != 0",
                            numerical_evidence=evidence,
                        )
                except (TypeError, ValueError, sympy.core.sympify.SympifyError):
                    continue

            return _CheckResult(
                status=FilterStatus.SURVIVING,
                numerical_evidence=evidence,
            )

        return None

    def _check_modular_pattern(
        self,
        conjecture: Conjecture,
        stmt: str,
        evidence: dict,
    ) -> _CheckResult | None:
        """Check modular arithmetic statements numerically.

        Handles patterns like "f(n) mod m = c" or "f(n) % m = c".

        Args:
            conjecture: The conjecture being checked.
            stmt: Lowercase statement text.
            evidence: Mutable dict to append test results into.

        Returns:
            A _CheckResult if a pattern was matched, None otherwise.
        """
        import sympy  # noqa: PLC0415

        mod_patterns = [
            r"(.+?)\s*(?:mod|%)\s*(\d+)\s*=\s*(\d+)",
            r"(.+?)\s+modulo\s+(\d+)\s+(?:is|equals?)\s+(\d+)",
        ]
        for pat in mod_patterns:
            m = re.search(pat, stmt)
            if m is None:
                continue

            expr_str, mod_str, rem_str = m.group(1), m.group(2), m.group(3)

            try:
                mod_k = int(mod_str.strip())
                rem = int(rem_str.strip())
            except ValueError:
                continue

            if mod_k <= 0 or rem < 0 or rem >= mod_k:
                continue

            n_sym = sympy.Symbol("n", positive=True, integer=True)
            p_sym = sympy.Symbol("p", positive=True, integer=True)

            try:
                expr = sympy.sympify(
                    expr_str.strip(),
                    locals={"n": n_sym, "p": p_sym},
                )
            except (sympy.SympifyError, SyntaxError):
                continue

            test_vals = self._generate_integer_test_values()
            for val in test_vals:
                try:
                    result = int(expr.subs(n_sym, val).evalf())
                    actual_rem = result % mod_k
                    passes = actual_rem == rem
                    evidence[f"n={val}"] = passes
                    if not passes:
                        return _CheckResult(
                            status=FilterStatus.DISPROVED,
                            counterexample=(
                                f"n={val}: {result} mod {mod_k} = {actual_rem} != {rem}"
                            ),
                            numerical_evidence=evidence,
                        )
                except (TypeError, ValueError):
                    continue

            return _CheckResult(
                status=FilterStatus.SURVIVING,
                numerical_evidence=evidence,
            )

        return None

    def _check_gcd_pattern(
        self,
        conjecture: Conjecture,
        stmt: str,
        evidence: dict,
    ) -> _CheckResult | None:
        """Check GCD equality statements numerically.

        Handles patterns like "gcd(f(n), g(n)) = k".

        Args:
            conjecture: The conjecture being checked.
            stmt: Lowercase statement text.
            evidence: Mutable dict to append test results into.

        Returns:
            A _CheckResult if a pattern was matched, None otherwise.
        """
        import sympy  # noqa: PLC0415
        from math import gcd as math_gcd  # noqa: PLC0415

        gcd_pattern = r"gcd\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*=\s*(\d+)"
        m = re.search(gcd_pattern, stmt)
        if m is None:
            return None

        expr1_str, expr2_str, k_str = m.group(1), m.group(2), m.group(3)
        try:
            k = int(k_str.strip())
        except ValueError:
            return None

        n_sym = sympy.Symbol("n", positive=True, integer=True)
        try:
            expr1 = sympy.sympify(
                expr1_str.strip(),
                locals={"n": n_sym, "factorial": sympy.factorial},
            )
            expr2 = sympy.sympify(
                expr2_str.strip(),
                locals={"n": n_sym, "factorial": sympy.factorial},
            )
        except (sympy.SympifyError, SyntaxError):
            return None

        test_vals = self._generate_integer_test_values()
        for val in test_vals:
            try:
                v1 = int(expr1.subs(n_sym, val).evalf())
                v2 = int(expr2.subs(n_sym, val).evalf())
                actual_gcd = math_gcd(abs(v1), abs(v2))
                passes = actual_gcd == k
                evidence[f"n={val}"] = passes
                if not passes:
                    return _CheckResult(
                        status=FilterStatus.DISPROVED,
                        counterexample=f"n={val}: gcd({v1}, {v2}) = {actual_gcd} != {k}",
                        numerical_evidence=evidence,
                    )
            except (TypeError, ValueError):
                continue

        return _CheckResult(
            status=FilterStatus.SURVIVING,
            numerical_evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Inequality
    # ------------------------------------------------------------------

    def check_inequality(self, conjecture: Conjecture) -> _CheckResult:
        """Run inequality-specific falsification checks using random sampling.

        Randomly samples (a, b, c) from (0, 10] and evaluates whether the
        stated inequality holds. Also tests edge cases (1, 1, 1) and near-zero
        (0.001, 0.001, 0.001).

        Args:
            conjecture: The inequality conjecture to check.

        Returns:
            A _CheckResult with the outcome.
        """
        import sympy  # noqa: PLC0415

        stmt = conjecture.nl_statement
        stmt_lower = stmt.lower()

        # --- Trivial check (a >= 0 for Nat, or 0 = 0, etc.) ---
        if self._is_trivial_statement(stmt_lower):
            return _CheckResult(status=FilterStatus.TRIVIAL)

        # Extract the inequality expression using heuristic pattern matching
        ineq_result = self._check_inequality_expression(stmt, stmt_lower)
        if ineq_result is not None:
            return ineq_result

        return _CheckResult(status=FilterStatus.SURVIVING)

    def _check_inequality_expression(
        self,
        stmt: str,
        stmt_lower: str,
    ) -> _CheckResult | None:
        """Attempt to parse and numerically test an inequality.

        Handles patterns like "f(a, b) >= g(a, b)" and "f(a, b) <= g(a, b)".

        Args:
            stmt: Original statement text.
            stmt_lower: Lowercase version.

        Returns:
            A _CheckResult if a testable inequality was found, None otherwise.
        """
        import sympy  # noqa: PLC0415

        # Match "lhs >= rhs" or "lhs <= rhs" (handles both >= and ≥)
        ineq_patterns = [
            (r"(.+?)\s*>=\s*(.+?)(?:\s+for|\s*$)", ">="),
            (r"(.+?)\s*<=\s*(.+?)(?:\s+for|\s*$)", "<="),
            (r"(.+?)\s*>\s*(.+?)(?:\s+for|\s*$)", ">"),
            (r"(.+?)\s*<\s*(.+?)(?:\s+for|\s*$)", "<"),
        ]

        a_sym = sympy.Symbol("a", positive=True, real=True)
        b_sym = sympy.Symbol("b", positive=True, real=True)
        c_sym = sympy.Symbol("c", positive=True, real=True)
        sym_locals = {"a": a_sym, "b": b_sym, "c": c_sym}

        for pat, op in ineq_patterns:
            m = re.search(pat, stmt_lower)
            if m is None:
                continue

            lhs_str = m.group(1).strip().rstrip(":")
            rhs_str = m.group(2).strip().rstrip(".")

            # Skip if either side contains words that suggest this is a
            # meta-description rather than a mathematical expression
            if any(
                word in lhs_str + rhs_str
                for word in ["for all", "there exist", "such that", "where"]
            ):
                continue

            try:
                lhs = sympy.sympify(lhs_str, locals=sym_locals)
                rhs = sympy.sympify(rhs_str, locals=sym_locals)
            except (sympy.SympifyError, SyntaxError):
                continue

            evidence: dict[str, bool] = {}
            test_points = self._generate_real_test_values()

            for point in test_points:
                a_val, b_val, c_val = point["a"], point["b"], point["c"]
                subs_map = {a_sym: a_val, b_sym: b_val, c_sym: c_val}
                try:
                    lhs_val = float(lhs.subs(subs_map).evalf())
                    rhs_val = float(rhs.subs(subs_map).evalf())
                except (TypeError, ValueError, ZeroDivisionError):
                    continue

                if not (lhs_val == lhs_val) or not (rhs_val == rhs_val):
                    # NaN — skip this point
                    continue

                passes = self._compare_values(lhs_val, rhs_val, op)
                key = f"a={a_val:.3f},b={b_val:.3f},c={c_val:.3f}"
                evidence[key] = passes

                if not passes:
                    return _CheckResult(
                        status=FilterStatus.DISPROVED,
                        counterexample=(
                            f"a={a_val:.4f}, b={b_val:.4f}, c={c_val:.4f}: "
                            f"lhs={lhs_val:.6f} {op} rhs={rhs_val:.6f} is False"
                        ),
                        numerical_evidence=evidence,
                    )

            return _CheckResult(
                status=FilterStatus.SURVIVING,
                numerical_evidence=evidence,
            )

        return None

    @staticmethod
    def _compare_values(lhs: float, rhs: float, op: str) -> bool:
        """Compare two float values with a small epsilon for numerical stability.

        Args:
            lhs: Left-hand side value.
            rhs: Right-hand side value.
            op: One of ">=", "<=", ">", "<".

        Returns:
            True if the comparison holds within numerical tolerance.
        """
        eps = 1e-9
        if op == ">=":
            return lhs >= rhs - eps
        elif op == "<=":
            return lhs <= rhs + eps
        elif op == ">":
            return lhs > rhs - eps
        elif op == "<":
            return lhs < rhs + eps
        return True

    # ------------------------------------------------------------------
    # Combinatorics
    # ------------------------------------------------------------------

    def check_combinatorics(self, conjecture: Conjecture) -> _CheckResult:
        """Run combinatorics-specific falsification checks.

        For bounded finite claims (n < 20), attempts to verify or disprove
        by direct enumeration. Unbounded claims are passed through.

        Args:
            conjecture: The combinatorics conjecture to check.

        Returns:
            A _CheckResult with the outcome.
        """
        import sympy  # noqa: PLC0415

        stmt = conjecture.nl_statement.lower()

        # Extract explicit bound if present
        bound_match = re.search(r"n\s*<\s*(\d+)|n\s*<=?\s*(\d+)", stmt)
        if bound_match is None:
            # No bound found — conservative pass
            return _CheckResult(status=FilterStatus.SURVIVING)

        bound_str = bound_match.group(1) or bound_match.group(2)
        try:
            bound = int(bound_str)
        except ValueError:
            return _CheckResult(status=FilterStatus.SURVIVING)

        # Cap to prevent runaway computation
        bound = min(bound, 25)

        # Try to evaluate a binomial identity or sum claim
        result = self._check_combinatorics_bounded(conjecture.nl_statement, bound)
        return result

    def _check_combinatorics_bounded(self, stmt: str, bound: int) -> _CheckResult:
        """Check a bounded combinatorics claim by enumeration.

        Args:
            stmt: The original statement text.
            bound: The upper bound on n.

        Returns:
            A _CheckResult with outcome and evidence.
        """
        import sympy  # noqa: PLC0415

        stmt_lower = stmt.lower()
        n_sym = sympy.Symbol("n", nonnegative=True, integer=True)
        evidence: dict[str, bool] = {}

        # Pattern: "sum of first n ... = f(n)"
        sum_eq_patterns = [
            r"sum\s+of\s+.+?=\s*(.+?)(?:\s+for|\s*$)",
        ]

        # Pattern: binomial identity C(2n, n) op C(n, k)
        binom_pattern = r"c\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*=\s*(.+?)(?:\s+for|\s*$)"
        m = re.search(binom_pattern, stmt_lower)
        if m is not None:
            lhs_n_str, lhs_k_str, rhs_str = m.group(1), m.group(2), m.group(3)
            try:
                lhs_n = sympy.sympify(lhs_n_str.strip(), locals={"n": n_sym})
                lhs_k = sympy.sympify(lhs_k_str.strip(), locals={"n": n_sym})
                rhs = sympy.sympify(rhs_str.strip(), locals={"n": n_sym})
                for val in range(bound):
                    lhs_val = sympy.binomial(
                        int(lhs_n.subs(n_sym, val)),
                        int(lhs_k.subs(n_sym, val)),
                    )
                    rhs_val = int(rhs.subs(n_sym, val).evalf())
                    passes = int(lhs_val) == rhs_val
                    evidence[f"n={val}"] = passes
                    if not passes:
                        return _CheckResult(
                            status=FilterStatus.DISPROVED,
                            counterexample=f"n={val}: C(...)={lhs_val} != {rhs_val}",
                            numerical_evidence=evidence,
                        )
                return _CheckResult(
                    status=FilterStatus.SURVIVING,
                    numerical_evidence=evidence,
                )
            except (TypeError, ValueError, sympy.SympifyError):
                pass

        return _CheckResult(status=FilterStatus.SURVIVING, numerical_evidence=evidence)

    # ------------------------------------------------------------------
    # Trivial detection
    # ------------------------------------------------------------------

    def _is_trivial_statement(self, stmt_lower: str) -> bool:
        """Heuristically detect trivially true statements.

        Checks for structural patterns that indicate the claim holds by
        definition or elementary reflexivity/identity.

        Args:
            stmt_lower: Lowercase statement text.

        Returns:
            True if the statement appears trivial.
        """
        trivial_patterns = [
            r"\bn\s*=\s*n\b",           # n = n
            r"\b0\s*\|\s*0\b",           # 0 | 0
            r"\ba\s*>=\s*0\b",           # a >= 0 for natural
            r"\bn\s*>=\s*0\b",           # n >= 0
            r"\bx\s*\+\s*0\s*=\s*x\b",  # x + 0 = x
            r"\b0\s*\+\s*x\s*=\s*x\b",  # 0 + x = x
            r"\bx\s*\*\s*1\s*=\s*x\b",  # x * 1 = x
            r"\b1\s*\*\s*x\s*=\s*x\b",  # 1 * x = x
        ]
        return any(re.search(p, stmt_lower) is not None for p in trivial_patterns)

    # ------------------------------------------------------------------
    # Test value generators
    # ------------------------------------------------------------------

    def _generate_integer_test_values(self) -> list[int]:
        """Generate integer test values for number-theory checks.

        Returns a mix of systematic small values, edge cases, and random
        samples from [1, 1000].

        Returns:
            A list of integer test values.
        """
        systematic = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
        random_vals = random.sample(range(1, 1001), min(self.n_random_attempts, 40))
        return list(dict.fromkeys(systematic + random_vals))

    def _generate_real_test_values(self) -> list[dict[str, float]]:
        """Generate real-valued test points for inequality checks.

        Returns a mix of edge cases and random samples from (0, 10].

        Returns:
            A list of dicts with keys "a", "b", "c".
        """
        edge_cases = [
            {"a": 1.0, "b": 1.0, "c": 1.0},
            {"a": 0.001, "b": 0.001, "c": 0.001},
            {"a": 10.0, "b": 10.0, "c": 10.0},
            {"a": 1.0, "b": 2.0, "c": 3.0},
            {"a": 0.5, "b": 2.0, "c": 0.5},
            {"a": 0.01, "b": 9.99, "c": 1.0},
        ]
        random_vals = [
            {
                "a": random.uniform(1e-4, 10.0),
                "b": random.uniform(1e-4, 10.0),
                "c": random.uniform(1e-4, 10.0),
            }
            for _ in range(self.n_random_attempts)
        ]
        return edge_cases + random_vals


# ---------------------------------------------------------------------------
# Public SympyFilter class
# ---------------------------------------------------------------------------


class SympyFilter:
    """Conservative symbolic and numerical filter for mathematical conjectures.

    Applies SymPy-based falsification within a strict per-conjecture timeout
    enforced via multiprocessing. The filter is deliberately conservative:
    a conjecture is only classified as DISPROVED when a concrete counterexample
    is verified; all other outcomes default to SURVIVING.

    Attributes:
        n_test_values: Number of systematic test values to try per conjecture.
        n_random_attempts: Number of random test points to try per conjecture.
    """

    def __init__(
        self,
        n_test_values: int = 50,
        n_random_attempts: int = 200,
    ) -> None:
        """Initialize the filter with sampling parameters.

        Args:
            n_test_values: Number of systematic test values (e.g., n in 0..100).
            n_random_attempts: Number of random test points (random integers or
                real-valued samples).
        """
        if n_test_values < 1:
            raise ValueError(f"n_test_values must be >= 1, got {n_test_values}")
        if n_random_attempts < 1:
            raise ValueError(f"n_random_attempts must be >= 1, got {n_random_attempts}")
        self.n_test_values = n_test_values
        self.n_random_attempts = n_random_attempts
        logger.info(
            "SympyFilter initialized | n_test_values=%d | n_random_attempts=%d",
            n_test_values,
            n_random_attempts,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, conjecture: Conjecture) -> FilterResult:
        """Apply falsification logic to a single conjecture.

        Spawns a worker process with a hard timeout to prevent SymPy from
        hanging on complex expressions.

        Strategy:
        1. Parse the statement into a SymPy-checkable form via pattern matching.
        2. Test on n_test_values systematic values and n_random_attempts random values.
        3. If a counterexample is found -> DISPROVED.
        4. If all tests pass -> SURVIVING (with numerical evidence).
        5. If unparseable -> SURVIVING (conservative pass-through).

        Args:
            conjecture: The conjecture to evaluate.

        Returns:
            A FilterResult with status, optional counterexample, and evidence.
        """
        task = _CheckTask(
            conjecture=conjecture,
            n_test_values=self.n_test_values,
            n_random_attempts=self.n_random_attempts,
        )

        ctx = multiprocessing.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(processes=1) as pool:
            async_result = pool.apply_async(_worker_check, (task,))
            try:
                check_result: _CheckResult = async_result.get(
                    timeout=_SYMPY_TIMEOUT_SECONDS + 1
                )
            except multiprocessing.TimeoutError:
                logger.debug(
                    "Conjecture %s timed out in worker process — classifying as SURVIVING",
                    conjecture.id,
                )
                check_result = _CheckResult(
                    status=FilterStatus.SURVIVING,
                    error="worker timeout",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Worker process raised unexpected exception for conjecture %s: %s",
                    conjecture.id,
                    exc,
                )
                check_result = _CheckResult(
                    status=FilterStatus.SURVIVING,
                    error=str(exc),
                )

        if check_result.error:
            logger.debug(
                "Conjecture %s: worker error=%s, status=SURVIVING",
                conjecture.id,
                check_result.error,
            )

        return FilterResult(
            conjecture=conjecture,
            status=check_result.status,
            counterexample=check_result.counterexample,
            numerical_evidence=check_result.numerical_evidence,
        )

    def filter_batch(self, conjectures: list[Conjecture]) -> list[FilterResult]:
        """Filter a batch of conjectures with a tqdm progress bar.

        Processes conjectures sequentially (one subprocess per conjecture to
        enforce per-conjecture timeouts). Progress is tracked in-place via tqdm.

        Args:
            conjectures: List of conjectures to filter.

        Returns:
            List of FilterResult objects in the same order as the input.

        Raises:
            ValueError: If conjectures is empty.
        """
        if not conjectures:
            raise ValueError("conjectures list must not be empty")

        results: list[FilterResult] = []
        survived = 0
        disproved = 0
        trivial = 0

        with tqdm(
            conjectures,
            desc="Filtering",
            unit="conjecture",
            dynamic_ncols=True,
            postfix={"survived": 0, "disproved": 0, "trivial": 0},
        ) as pbar:
            for conjecture in pbar:
                result = self.filter(conjecture)
                results.append(result)

                if result.status == FilterStatus.SURVIVING:
                    survived += 1
                elif result.status == FilterStatus.DISPROVED:
                    disproved += 1
                elif result.status == FilterStatus.TRIVIAL:
                    trivial += 1

                pbar.set_postfix(
                    survived=survived,
                    disproved=disproved,
                    trivial=trivial,
                    refresh=False,
                )

        logger.info(
            "Batch filter complete: %d total | %d surviving | %d disproved | %d trivial",
            len(results),
            survived,
            disproved,
            trivial,
        )
        return results

    # ------------------------------------------------------------------
    # Delegated single-conjecture checks (thin wrappers for testability)
    # ------------------------------------------------------------------

    def _check_number_theory(self, conjecture: Conjecture) -> FilterResult:
        """Run number-theory checks synchronously in the current process.

        NOTE: This method does NOT enforce a timeout. Use ``filter()`` in
        production to get subprocess isolation.

        Args:
            conjecture: A number-theory conjecture.

        Returns:
            A FilterResult with outcome and evidence.
        """
        checker = _SympyCheckers(self.n_test_values, self.n_random_attempts)
        result = checker.check_number_theory(conjecture)
        return FilterResult(
            conjecture=conjecture,
            status=result.status,
            counterexample=result.counterexample,
            numerical_evidence=result.numerical_evidence,
        )

    def _check_inequality(self, conjecture: Conjecture) -> FilterResult:
        """Run inequality checks synchronously in the current process.

        NOTE: This method does NOT enforce a timeout. Use ``filter()`` in
        production to get subprocess isolation.

        Args:
            conjecture: An inequality conjecture.

        Returns:
            A FilterResult with outcome and evidence.
        """
        checker = _SympyCheckers(self.n_test_values, self.n_random_attempts)
        result = checker.check_inequality(conjecture)
        return FilterResult(
            conjecture=conjecture,
            status=result.status,
            counterexample=result.counterexample,
            numerical_evidence=result.numerical_evidence,
        )

    def _is_trivial(self, conjecture: Conjecture) -> bool:
        """Heuristically check if a conjecture is trivially true.

        Args:
            conjecture: The conjecture to inspect.

        Returns:
            True if the conjecture appears trivially true.
        """
        checker = _SympyCheckers(self.n_test_values, self.n_random_attempts)
        return checker._is_trivial_statement(conjecture.nl_statement.lower())

    def _generate_test_values(self, domain: Domain) -> list[dict]:
        """Generate test value dicts appropriate for the given domain.

        Args:
            domain: The mathematical domain.

        Returns:
            A list of test value dicts. For NUMBER_THEORY, each dict has
            key ``"n"``; for INEQUALITY and COMBINATORICS, each dict has
            keys ``"a"``, ``"b"``, ``"c"``.
        """
        checker = _SympyCheckers(self.n_test_values, self.n_random_attempts)
        if domain == Domain.NUMBER_THEORY:
            return [{"n": v} for v in checker._generate_integer_test_values()]
        else:
            return checker._generate_real_test_values()
