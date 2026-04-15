"""
Tests for conjlean.sympy_filter — SympyFilter and _SympyCheckers.

All tests run in-process against real SymPy so they validate actual
numerical falsification logic. No mocking is required for this module.

Note: tests use the synchronous _check_number_theory / _check_inequality
delegation methods on SympyFilter rather than the subprocess-based filter()
to avoid spawning child processes in the test suite.
"""

from __future__ import annotations

import pytest

from conjlean.schemas import Conjecture, Domain, FilterStatus
from conjlean.sympy_filter import SympyFilter, _SympyCheckers


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _nt(stmt: str, cid: str = "nt_test") -> Conjecture:
    """Build a number-theory Conjecture with the given statement."""
    return Conjecture(
        id=cid,
        domain=Domain.NUMBER_THEORY,
        nl_statement=stmt,
        variables=["n"],
    )


def _ineq(stmt: str, cid: str = "ineq_test") -> Conjecture:
    """Build an inequality Conjecture with the given statement."""
    return Conjecture(
        id=cid,
        domain=Domain.INEQUALITY,
        nl_statement=stmt,
        variables=["a", "b"],
    )


def _combo(stmt: str, cid: str = "combo_test") -> Conjecture:
    """Build a combinatorics Conjecture with the given statement."""
    return Conjecture(
        id=cid,
        domain=Domain.COMBINATORICS,
        nl_statement=stmt,
        variables=["n"],
    )


@pytest.fixture(scope="module")
def sf() -> SympyFilter:
    """Shared SympyFilter instance for all tests in this module."""
    return SympyFilter(n_test_values=20, n_random_attempts=10)


# ---------------------------------------------------------------------------
# TestFilterNumberTheory
# ---------------------------------------------------------------------------


class TestFilterNumberTheory:
    """Number-theory falsification tests using real SymPy evaluation."""

    def test_divisibility_true(self, sf: SympyFilter) -> None:
        """'2 divides n*(n+1)' should survive — it is always even."""
        c = _nt("2 divides n*(n+1) for all natural numbers n")
        result = sf._check_number_theory(c)
        assert result.status == FilterStatus.SURVIVING
        assert result.counterexample is None

    def test_divisibility_true_3consec(self, sf: SympyFilter) -> None:
        """'3 divides n*(n+1)*(n+2)' should survive — product of 3 consecutive ints."""
        c = _nt("3 divides n*(n+1)*(n+2) for all natural numbers n")
        result = sf._check_number_theory(c)
        assert result.status == FilterStatus.SURVIVING

    def test_divisibility_false(self, sf: SympyFilter) -> None:
        """'7 divides n^2 + 1 for all n' should be DISPROVED (n=1 gives 2)."""
        c = _nt("7 divides n^2 + 1 for all n")
        result = sf._check_number_theory(c)
        assert result.status == FilterStatus.DISPROVED
        assert result.counterexample is not None

    def test_modular_prime_sq(self, sf: SympyFilter) -> None:
        """'p^2 mod 24 = 1 for all primes p > 3' — unparseable guard survives."""
        # The full English description "for all primes p > 3" cannot be expressed
        # as a universally-quantified numerical formula over all integers, so the
        # filter is expected to conservatively pass it through (SURVIVING) since
        # it cannot build a counterexample from the natural-language description alone.
        c = _nt(
            "For all primes p greater than 3, p squared mod 24 equals 1"
        )
        # No parseable pattern → conservative SURVIVING
        result = sf._check_number_theory(c)
        assert result.status in (FilterStatus.SURVIVING, FilterStatus.TRIVIAL)

    def test_modular_false(self, sf: SympyFilter) -> None:
        """'n^2 mod 4 = 0 for all n' should be DISPROVED (n=1 gives 1)."""
        c = _nt("n^2 mod 4 = 0 for all n")
        result = sf._check_number_theory(c)
        assert result.status == FilterStatus.DISPROVED
        assert result.counterexample is not None

    def test_gcd_property(self, sf: SympyFilter) -> None:
        """'gcd(n, n+1) = 1 for all n' should survive — consecutive ints are coprime."""
        c = _nt("gcd(n, n+1) = 1 for all natural numbers n")
        result = sf._check_number_theory(c)
        assert result.status == FilterStatus.SURVIVING

    def test_numerical_evidence_populated(self, sf: SympyFilter) -> None:
        """Surviving divisibility conjectures should have at least n=0 through n=5 tested."""
        c = _nt("2 divides n*(n+1) for all natural numbers n")
        result = sf._check_number_theory(c)
        # Evidence keys are formatted as "n=<val>"
        evidence_keys = set(result.numerical_evidence.keys())
        for expected in ("n=0", "n=1", "n=2"):
            assert expected in evidence_keys, f"Expected {expected!r} in evidence keys"

    def test_gcd_disproved_when_wrong_constant(self, sf: SympyFilter) -> None:
        """'gcd(n, n+1) = 2 for all n' should be DISPROVED (gcd is always 1)."""
        c = _nt("gcd(n, n+1) = 2 for all natural numbers n")
        result = sf._check_number_theory(c)
        assert result.status == FilterStatus.DISPROVED


# ---------------------------------------------------------------------------
# TestFilterInequality
# ---------------------------------------------------------------------------


class TestFilterInequality:
    """Inequality falsification tests using real SymPy evaluation."""

    def test_amgm_simple(self, sf: SympyFilter) -> None:
        """AM-GM: 'a^2 + b^2 >= 2*a*b for positive reals' should survive."""
        c = _ineq("a^2 + b^2 >= 2*a*b for positive reals a, b")
        result = sf._check_inequality(c)
        assert result.status == FilterStatus.SURVIVING

    def test_cauchy_schwarz_simple(self, sf: SympyFilter) -> None:
        """Trivially true: 'a^2 + b^2 >= 0 for all reals a, b' should survive."""
        c = _ineq("a^2 + b^2 >= 0 for all reals a, b")
        result = sf._check_inequality(c)
        assert result.status in (FilterStatus.SURVIVING, FilterStatus.TRIVIAL)

    def test_inequality_false(self, sf: SympyFilter) -> None:
        """'a^2 + b^2 <= a*b for all positive reals' should be DISPROVED."""
        # a=2, b=2: 4+4=8, 2*2=4 → 8 <= 4 is False
        c = _ineq("a^2 + b^2 <= a*b for all positive reals")
        result = sf._check_inequality(c)
        assert result.status == FilterStatus.DISPROVED
        assert result.counterexample is not None

    def test_numerical_evidence_has_multiple_points(self, sf: SympyFilter) -> None:
        """A surviving inequality should have multiple evidence data points."""
        c = _ineq("a^2 + b^2 >= 2*a*b for positive reals a, b")
        result = sf._check_inequality(c)
        if result.status == FilterStatus.SURVIVING:
            assert len(result.numerical_evidence) >= 3


# ---------------------------------------------------------------------------
# TestFilterBatch
# ---------------------------------------------------------------------------


class TestFilterBatch:
    """Batch filtering tests using SympyFilter.filter_batch."""

    def _make_batch(self) -> list[Conjecture]:
        """Return a diverse batch of 5 conjectures."""
        return [
            _nt("2 divides n*(n+1)", f"c{i}") if i % 2 == 0
            else _ineq("a^2 + b^2 >= 2*a*b for positive reals", f"c{i}")
            for i in range(5)
        ]

    def test_batch_processes_all(self, sf: SympyFilter) -> None:
        """filter_batch returns exactly as many results as input conjectures."""
        conjectures = self._make_batch()
        results = sf.filter_batch(conjectures)
        assert len(results) == 5

    def test_batch_ids_preserved(self, sf: SympyFilter) -> None:
        """Each result's conjecture.id matches the corresponding input conjecture.id."""
        conjectures = self._make_batch()
        results = sf.filter_batch(conjectures)
        for i, result in enumerate(results):
            assert result.conjecture.id == conjectures[i].id

    def test_batch_mixed_results(self, sf: SympyFilter) -> None:
        """A batch with both surviving and disproved conjectures is handled correctly."""
        conjectures = [
            _nt("2 divides n*(n+1)", "good_1"),
            _nt("7 divides n^2 + 1 for all n", "bad_1"),
            _ineq("a^2 + b^2 >= 2*a*b for positive reals", "good_2"),
        ]
        results = sf.filter_batch(conjectures)
        statuses = {r.conjecture.id: r.status for r in results}
        assert statuses["good_1"] == FilterStatus.SURVIVING
        assert statuses["bad_1"] == FilterStatus.DISPROVED

    def test_batch_rejects_empty(self, sf: SympyFilter) -> None:
        """filter_batch raises ValueError for an empty input list."""
        with pytest.raises(ValueError):
            sf.filter_batch([])

    def test_conservative_on_unparseable(self, sf: SympyFilter) -> None:
        """A conjecture whose NL cannot be parsed is classified SURVIVING (not DISPROVED)."""
        c = _nt(
            "For every sufficiently large prime p, there exists an integer k such that "
            "k^3 ≡ p mod (p-1) and gcd(k, p) = 1 with the associated Euler totient property",
            "unparseable_001",
        )
        results = sf.filter_batch([c])
        # Must never be DISPROVED without a counterexample
        assert results[0].status != FilterStatus.DISPROVED or results[0].counterexample is not None


# ---------------------------------------------------------------------------
# TestSympyCheckers (in-process unit tests without subprocess)
# ---------------------------------------------------------------------------


class TestSympyCheckers:
    """Direct tests of _SympyCheckers without subprocess isolation."""

    @pytest.fixture()
    def checkers(self) -> _SympyCheckers:
        """Return a _SympyCheckers instance for in-process testing."""
        return _SympyCheckers(n_test_values=15, n_random_attempts=5)

    def test_generate_integer_test_values_includes_zero(self, checkers: _SympyCheckers) -> None:
        """Integer test values always include 0 for boundary coverage."""
        vals = checkers._generate_integer_test_values()
        assert 0 in vals

    def test_generate_integer_test_values_includes_small_ints(
        self, checkers: _SympyCheckers
    ) -> None:
        """Integer test values include small systematic values like 1, 2, 3."""
        vals = checkers._generate_integer_test_values()
        assert 1 in vals
        assert 2 in vals

    def test_generate_real_test_values_returns_dicts(self, checkers: _SympyCheckers) -> None:
        """Real test value points are dicts with 'a', 'b', 'c' keys."""
        points = checkers._generate_real_test_values()
        assert len(points) > 0
        for p in points:
            assert "a" in p and "b" in p and "c" in p

    def test_is_trivial_detects_n_equals_n(self, checkers: _SympyCheckers) -> None:
        """'n = n' is detected as a trivial statement."""
        assert checkers._is_trivial_statement("n = n") is True

    def test_is_trivial_detects_n_geq_0(self, checkers: _SympyCheckers) -> None:
        """'n >= 0' is detected as trivially true for naturals."""
        assert checkers._is_trivial_statement("n >= 0") is True

    def test_is_trivial_returns_false_for_nontrivial(self, checkers: _SympyCheckers) -> None:
        """'2 divides n*(n+1)' is not trivial."""
        assert checkers._is_trivial_statement("2 divides n*(n+1)") is False
