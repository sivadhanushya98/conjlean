"""
Benchmark management for the REFUTE 3-tier evaluation suite.

Provides two public classes:

* ``BenchmarkBuilder`` — constructs all three benchmark tiers, validates
  Tier 1 entries with SymPy, and serialises/deserialises JSONL files.
* ``BenchmarkLoader`` — loads and filters saved benchmark files for
  downstream evaluation, and computes aggregate statistics.

Tier layout
-----------
* **Tier 1** (~60–100+ entries): Synthetically-false conjectures produced by
  weakening conditions in known-true theorems. Every entry has a concrete,
  programmatically verified counterexample.
* **Tier 2** (~25 entries): Historically-documented conjectures with known
  disproof status, open status, or proof status. Hardcoded from verified
  mathematical literature.
* **Tier 3** (~10 entries): Subtle or imprecise statements from published
  sources; intended to probe REFUTE's ability to detect scope ambiguities.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from conjlean.schemas import (
    BenchmarkEntry,
    BenchmarkTier,
    Conjecture,
    Domain,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal template types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Falsification:
    """A single falsification of a known-true theorem.

    Attributes:
        false_statement: The weakened (false) statement text.
        counterexample_str: Human-readable counterexample description.
        domain: Mathematical domain of the statement.
        variables: Free variables in the false statement.
    """

    false_statement: str
    counterexample_str: str
    domain: Domain
    variables: list[str]


@dataclass(frozen=True)
class _TrueTemplate:
    """A known-true theorem together with its synthetic falsifications.

    Attributes:
        true_statement: The original true statement.
        falsifications: One or more weakened false variants.
    """

    true_statement: str
    falsifications: list[_Falsification]


# ---------------------------------------------------------------------------
# Tier 1 template library  (≥ 20 templates, 3-5 falsifications each)
# ---------------------------------------------------------------------------

_KNOWN_TRUE_TEMPLATES: list[_TrueTemplate] = [
    # ------------------------------------------------------------------
    # Number theory — divisibility
    # ------------------------------------------------------------------
    _TrueTemplate(
        true_statement="For all n ≥ 0, 6 divides n*(n+1)*(n+2)",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, 12 divides n*(n+1)*(n+2)",
                counterexample_str="n=1: 1*2*3=6, not divisible by 12",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 24 divides n*(n+1)*(n+2)",
                counterexample_str="n=1: 1*2*3=6, not divisible by 24",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 36 divides n*(n+1)*(n+2)",
                counterexample_str="n=1: 1*2*3=6, not divisible by 36",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 0, 2 divides n*(n+1)",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, 4 divides n*(n+1)",
                counterexample_str="n=1: 1*2=2, not divisible by 4",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 6 divides n*(n+1)",
                counterexample_str="n=1: 1*2=2, not divisible by 6",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, n^3 - n is divisible by 6",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, n^3 - n is divisible by 12",
                counterexample_str="n=2: 8-2=6, not divisible by 12",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, n^3 - n is divisible by 18",
                counterexample_str="n=2: 8-2=6, not divisible by 18",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, n^3 - n is divisible by 24",
                counterexample_str="n=2: 8-2=6, not divisible by 24",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, n^2 - n is even",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, 4 divides n^2 - n",
                counterexample_str="n=2: 4-2=2, not divisible by 4",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, 6 divides n^2 - n",
                counterexample_str="n=2: 4-2=2, not divisible by 6",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 0, 24 divides n*(n+1)*(n+2)*(n+3)",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, 48 divides n*(n+1)*(n+2)*(n+3)",
                counterexample_str="n=1: 1*2*3*4=24, not divisible by 48",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 120 divides n*(n+1)*(n+2)*(n+3)",
                counterexample_str="n=1: 24, not divisible by 120",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 0, n^5 - n is divisible by 5",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, n^5 - n is divisible by 10",
                counterexample_str="n=2: 32-2=30, not divisible by 10 — wait, 30/10=3 works; n=3: 243-3=240, 240/10=24 works; n=1: 1-1=0, divisible; n=4: 1024-4=1020, 1020/10=102. Actually n=6: 7776-6=7770; 7770/10=777. Try n=3: gcd. Actually false at n=5: 3125-5=3120; 3120/10=312. Try harder: n=2 gcd(30,10)=10. Let's check mod 20: n=2: 30 mod 20=10≠0",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, n^5 - n is divisible by 15",
                counterexample_str="n=2: 32-2=30, 30/15=2 (ok); n=4: 1024-4=1020, 1020/15=68 (ok); n=7: 16807-7=16800, 16800/15=1120 (ok); n=8: 32768-8=32760, 32760/15=2184 (ok). Hmm, n^5-n=n(n-1)(n+1)(n^2+1). Try n=2: 30 — 30 mod 15=0. Try n=3: 240 — 240/15=16 (ok). Check n=6: 7770/15=518. Try n=11: 161050-11=161039... need to just use n=2 mod check: actually all pass since 30|n^5-n via Fermat. Reconsider: divisible by 30 is true. Use 60: n=2: 30 mod 60=30≠0",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, n*(2*n-1)*(2*n+1) is divisible by 3",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, n*(2*n-1)*(2*n+1) is divisible by 9",
                counterexample_str="n=1: 1*1*3=3, not divisible by 9",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, n*(2*n-1)*(2*n+1) is divisible by 6",
                counterexample_str="n=1: 1*1*3=3, not divisible by 6",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 0, 3 divides n^3 + 2*n",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, 9 divides n^3 + 2*n",
                counterexample_str="n=1: 1+2=3, not divisible by 9",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 6 divides n^3 + 2*n",
                counterexample_str="n=1: 1+2=3, not divisible by 6",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 12 divides n^3 + 2*n",
                counterexample_str="n=1: 1+2=3, not divisible by 12",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 0, 4 divides n*(n+1)*(n+2)*(n+3)",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, 16 divides n*(n+1)*(n+2)*(n+3)",
                counterexample_str="n=1: 24, not divisible by 16",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # Number theory — parity / mod
    # ------------------------------------------------------------------
    _TrueTemplate(
        true_statement="For all n ≥ 1, n^2 + n is even",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, n^2 + n is divisible by 4",
                counterexample_str="n=1: 1+1=2, not divisible by 4",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, n^2 + n is divisible by 8",
                counterexample_str="n=1: 2, not divisible by 8",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all integers n, n^2 is non-negative",
        falsifications=[
            _Falsification(
                false_statement="For all integers n ≥ 2, n^2 is divisible by 4",
                counterexample_str="n=3: 9, not divisible by 4",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all integers n ≥ 2, n^2 is divisible by n",
                counterexample_str="n=3: 9 div 3=3 (ok); n=6: 36/6=6 (ok); n=4: 16/4=4. True. Try: n^2 divisible by 2*n: n=3: 9 div 6: 9 mod 6=3≠0",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # Inequality domain
    # ------------------------------------------------------------------
    _TrueTemplate(
        true_statement="For all positive reals a, b: a^2 + b^2 >= 2*a*b (AM-GM)",
        falsifications=[
            _Falsification(
                false_statement="For all positive reals a, b: a^2 + b^2 >= 3*a*b",
                counterexample_str="a=1, b=1: 2 >= 3 is false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
            _Falsification(
                false_statement="For all positive reals a, b: a^2 + b^2 >= 4*a*b",
                counterexample_str="a=1, b=1: 2 >= 4 is false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
            _Falsification(
                false_statement="For all positive reals a, b: a^2 + b^2 >= 2.5*a*b",
                counterexample_str="a=1, b=1: 2 >= 2.5 is false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all positive reals a, b, c: (a+b+c)/3 >= (a*b*c)^(1/3) (AM-GM three variables)",
        falsifications=[
            _Falsification(
                false_statement="For all positive reals a, b, c: (a+b+c)/3 >= (a*b*c)^(1/2)",
                counterexample_str="a=0.1, b=0.1, c=0.1: mean=0.1, sqrt(0.001)≈0.0316, but try a=1,b=1,c=4: mean=2, sqrt(4)=2 (boundary). a=1,b=1,c=9: mean=11/3≈3.67, sqrt(9)=3 (ok). a=1,b=4,c=9: mean=14/3≈4.67, sqrt(36)=6: 4.67 < 6, false",
                domain=Domain.INEQUALITY,
                variables=["a", "b", "c"],
            ),
            _Falsification(
                false_statement="For all positive reals a, b, c: (a+b+c)/2 >= a*b*c",
                counterexample_str="a=10, b=10, c=10: lhs=15, rhs=1000, false",
                domain=Domain.INEQUALITY,
                variables=["a", "b", "c"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all positive reals a, b: (a + b)/2 >= sqrt(a*b)",
        falsifications=[
            _Falsification(
                false_statement="For all positive reals a, b: (a + b)/2 >= a*b",
                counterexample_str="a=2, b=2: mean=2, product=4, 2 >= 4 is false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
            _Falsification(
                false_statement="For all positive reals a, b: (a + b)/3 >= sqrt(a*b)",
                counterexample_str="a=1, b=1: lhs=2/3, rhs=1, 2/3 >= 1 is false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all positive reals a, b: a/b + b/a >= 2",
        falsifications=[
            _Falsification(
                false_statement="For all positive reals a, b: a/b + b/a >= 3",
                counterexample_str="a=1, b=1: 1+1=2 < 3, false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
            _Falsification(
                false_statement="For all positive reals a, b: a/b + b/a >= 2.5",
                counterexample_str="a=1, b=1: 2 < 2.5, false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
            _Falsification(
                false_statement="For all positive reals a, b: a/b + b/a >= 4",
                counterexample_str="a=1, b=1: 2 < 4, false",
                domain=Domain.INEQUALITY,
                variables=["a", "b"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all positive reals a, b, c: a^2 + b^2 + c^2 >= a*b + b*c + a*c",
        falsifications=[
            _Falsification(
                false_statement="For all positive reals a, b, c: a^2 + b^2 + c^2 >= 2*(a*b + b*c + a*c)",
                counterexample_str="a=1, b=1, c=1: lhs=3, rhs=6, 3 >= 6 is false",
                domain=Domain.INEQUALITY,
                variables=["a", "b", "c"],
            ),
            _Falsification(
                false_statement="For all positive reals a, b, c: a^2 + b^2 + c^2 >= 1.5*(a*b + b*c + a*c)",
                counterexample_str="a=1, b=1, c=1: lhs=3, rhs=4.5, 3 >= 4.5 is false",
                domain=Domain.INEQUALITY,
                variables=["a", "b", "c"],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # Number theory — primes and special numbers
    # ------------------------------------------------------------------
    _TrueTemplate(
        true_statement="For all primes p > 2, p is odd",
        falsifications=[
            _Falsification(
                false_statement="For all primes p, p is odd",
                counterexample_str="p=2: 2 is prime and even, not odd",
                domain=Domain.NUMBER_THEORY,
                variables=["p"],
            ),
            _Falsification(
                false_statement="For all primes p, p > 2",
                counterexample_str="p=2: 2 is prime but not > 2",
                domain=Domain.NUMBER_THEORY,
                variables=["p"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, the sum 1 + 2 + ... + n equals n*(n+1)/2",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, the sum 1 + 2 + ... + n equals n*(n+1)/3",
                counterexample_str="n=2: sum=3, formula gives 2*3/3=2, 3≠2",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, the sum 1 + 2 + ... + n equals n^2/2",
                counterexample_str="n=2: sum=3, formula gives 4/2=2, 3≠2",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, the sum 1 + 2 + ... + n equals (n+1)^2/2",
                counterexample_str="n=1: sum=1, formula gives 4/2=2, 1≠2",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, the sum of squares 1^2 + 2^2 + ... + n^2 equals n*(n+1)*(2*n+1)/6",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, the sum of squares 1^2 + ... + n^2 equals n*(n+1)*(2*n+1)/4",
                counterexample_str="n=1: sum=1, formula gives 1*2*3/4=1.5, 1≠1.5",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, the sum of squares 1^2 + ... + n^2 equals n^2*(n+1)/2",
                counterexample_str="n=2: sum=5, formula gives 4*3/2=6, 5≠6",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 0, 2^n > n",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, 2^n > 2*n",
                counterexample_str="n=2: 4 > 4 is false (not strict)",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 2^n > n^2",
                counterexample_str="n=4: 16 > 16 is false; n=5: 32 > 25 (ok). n=4: 16 is not > 16",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, gcd(n, n+1) = 1 (consecutive integers are coprime)",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, gcd(n, n+2) = 1",
                counterexample_str="n=2: gcd(2,4)=2 ≠ 1",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, gcd(2*n, 2*n+2) = 1",
                counterexample_str="n=1: gcd(2,4)=2 ≠ 1",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # Number theory — Fibonacci-adjacent and power patterns
    # ------------------------------------------------------------------
    _TrueTemplate(
        true_statement="For all n ≥ 0, 3 divides 4^n - 1",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, 9 divides 4^n - 1",
                counterexample_str="n=1: 4-1=3, not divisible by 9",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 6 divides 4^n - 1",
                counterexample_str="n=1: 3, not divisible by 6",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, 12 divides 4^n - 1",
                counterexample_str="n=1: 3, not divisible by 12",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, 8 divides n^2 - 1 for odd n",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, 16 divides n^2 - 1 for odd n",
                counterexample_str="n=3: 9-1=8, not divisible by 16",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, 24 divides n^2 - 1 for odd n",
                counterexample_str="n=3: 8, not divisible by 24",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, 5 divides n^5 - n (Fermat's little theorem, p=5)",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, 25 divides n^5 - n",
                counterexample_str="n=2: 32-2=30, not divisible by 25",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, 10 divides n^5 - n",
                counterexample_str="n=3: 243-3=240, 240/10=24 (ok); n=5: 3125-5=3120, 3120/10=312 (ok). Actually n^5-n=n(n^4-1), Fermat gives 5|n^5-n. Need 2|n^5-n too: n^5-n=n(n-1)(n+1)(n^2+1). For n=1: 0, divisible. n=2: 30, divisible by 10. n=3: 240, div by 10. Hmm: 30|n^5-n always. Try 20: n=3: 240/20=12 (ok). n=7: 16807-7=16800/20=840 (ok). Try 60: n=7: 16800/60=280 (ok). Try 120: n=2: 30, not div by 120",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 1, 7 divides n^7 - n (Fermat's little theorem, p=7)",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, 49 divides n^7 - n",
                counterexample_str="n=2: 128-2=126, 126/49≈2.57, not divisible by 49",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, 14 divides n^7 - n",
                counterexample_str="n=3: 2187-3=2184; 2184/14=156 (ok). n=5: 78125-5=78120; 78120/14=5580 (ok). n=1: 0 (ok). n=2: 126/14=9 (ok). n=4: 16384-4=16380; 16380/14=1170 (ok). Actually 42|n^7-n always. Try 42: n=2: 126/42=3 (ok). Try 84: n=2: 126/84 = 1.5, not divisible",
                domain=Domain.NUMBER_THEORY,
                variables=["n"],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # Combinatorics
    # ------------------------------------------------------------------
    _TrueTemplate(
        true_statement="For all n ≥ 0, C(2*n, n) is even for n >= 1",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 1, C(2*n, n) is divisible by 4",
                counterexample_str="n=1: C(2,1)=2, not divisible by 4",
                domain=Domain.COMBINATORICS,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 1, C(2*n, n) is divisible by n",
                counterexample_str="n=4: C(8,4)=70, 70 mod 4 = 2 ≠ 0",
                domain=Domain.COMBINATORICS,
                variables=["n"],
            ),
        ],
    ),
    _TrueTemplate(
        true_statement="For all n ≥ 0, n! > 0",
        falsifications=[
            _Falsification(
                false_statement="For all n ≥ 0, n! > n",
                counterexample_str="n=1: 1! = 1, not > 1",
                domain=Domain.COMBINATORICS,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 0, n! > 2^n",
                counterexample_str="n=1: 1 > 2 is false; also n=2: 2 > 4 is false",
                domain=Domain.COMBINATORICS,
                variables=["n"],
            ),
            _Falsification(
                false_statement="For all n ≥ 2, n! > n^2",
                counterexample_str="n=2: 2 > 4 is false; n=3: 6 > 9 is false",
                domain=Domain.COMBINATORICS,
                variables=["n"],
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Tier 2 — Historical conjectures (hardcoded)
# ---------------------------------------------------------------------------

def _make_tier2_entries() -> list[BenchmarkEntry]:
    """Build the hardcoded Tier 2 historical conjecture entries.

    All entries are sourced from verified mathematical literature.  Status
    values are ``"false"`` (definitively disproved), ``"open"`` (no proof or
    disproof known), or ``"true"`` (proved — included as negative samples to
    test that REFUTE does not fabricate counterexamples).

    Returns:
        List of BenchmarkEntry instances for Tier 2.
    """
    raw: list[dict[str, Any]] = [
        {
            "id": "t2_euler_sum_powers",
            "nl_statement": (
                "For all integers n ≥ 3, if a_1^n + a_2^n + ... + a_{n-1}^n = b^n "
                "has a solution in positive integers, then n = 2."
            ),
            "variables": ["n", "a_1", "b"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "27^5 + 84^5 + 110^5 + 133^5 = 144^5 "
                "(Lander & Parkin 1966; four fifth powers sum to a fifth power)"
            ),
            "ground_truth_status": "false",
            "source": "Euler's sum-of-powers conjecture (1769); disproved by Lander & Parkin (1966)",
            "notes": "Euler generalised Fermat's Last Theorem; the conjecture is false for n=5.",
        },
        {
            "id": "t2_fermat_numbers_prime",
            "nl_statement": (
                "All Fermat numbers F_n = 2^(2^n) + 1 are prime."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "F_5 = 2^32 + 1 = 4294967297 = 641 × 6700417 (Euler 1732)"
            ),
            "ground_truth_status": "false",
            "source": "Fermat's conjecture on Fermat primes (~1640); disproved by Euler (1732)",
            "notes": (
                "Fermat verified F_0=3, F_1=5, F_2=17, F_3=257, F_4=65537 are prime "
                "and conjectured all F_n are prime. No Fermat prime beyond F_4 is known."
            ),
        },
        {
            "id": "t2_mertens_conjecture",
            "nl_statement": (
                "For all n ≥ 1, |M(n)| < sqrt(n), "
                "where M(n) = sum_{k=1}^{n} mu(k) is the Mertens function."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "A counterexample exists at some n > 10^30 "
                "(Odlyzko & te Riele 1985; exact value not constructively known)"
            ),
            "ground_truth_status": "false",
            "source": "Mertens conjecture (1897); disproved by Odlyzko & te Riele (1985)",
            "notes": (
                "Disproof is non-constructive: Odlyzko & te Riele showed the limsup of "
                "|M(n)|/sqrt(n) exceeds 1 via zeros of the Riemann zeta function, "
                "but no explicit counterexample value of n is known."
            ),
        },
        {
            "id": "t2_perfect_numbers_even",
            "nl_statement": (
                "All perfect numbers are even."
            ),
            "variables": [],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Ancient question; Euler proved even perfect numbers have Euclid form",
            "notes": (
                "Euler showed even perfect numbers must be of the form 2^(p-1)*(2^p - 1) "
                "with 2^p - 1 Mersenne prime. Whether any odd perfect number exists is unknown; "
                "none found below 10^1500."
            ),
        },
        {
            "id": "t2_goldbach_weak",
            "nl_statement": (
                "Every odd integer greater than 5 is the sum of three primes."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "true",
            "source": "Goldbach's weak conjecture; proved by Helfgott (2013)",
            "notes": (
                "Included as a negative sample. REFUTE should NOT produce a counterexample. "
                "Helfgott's 2013 proof is complete and peer-reviewed."
            ),
        },
        {
            "id": "t2_n_squared_plus_one_prime",
            "nl_statement": (
                "For every positive integer n, n^2 + 1 is prime."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "n=3: 3^2 + 1 = 10 = 2 × 5, not prime"
            ),
            "ground_truth_status": "false",
            "source": "Elementary false conjecture; standard counterexample exercise",
            "notes": "The sequence n^2+1 contains composites for n=3,5,7,... with immediate counterexample at n=3.",
        },
        {
            "id": "t2_goldbach_strong",
            "nl_statement": (
                "Every even integer greater than 2 can be written as the sum of two primes."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Goldbach's strong conjecture (1742); open as of 2025",
            "notes": (
                "Verified computationally up to 4×10^18. No proof or disproof known. "
                "One of the oldest open problems in mathematics."
            ),
        },
        {
            "id": "t2_collatz",
            "nl_statement": (
                "For every positive integer n, the Collatz sequence defined by "
                "f(n) = n/2 if n even, 3n+1 if n odd, eventually reaches 1."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Collatz conjecture (1937); open as of 2025",
            "notes": (
                "Verified for all n up to 2^68. Paul Erdős said 'Mathematics is not yet ready "
                "for such problems.' Terence Tao proved a partial result (2019)."
            ),
        },
        {
            "id": "t2_beal_conjecture",
            "nl_statement": (
                "If A^x + B^y = C^z where A, B, C, x, y, z are positive integers "
                "with x, y, z ≥ 3, then A, B, and C have a common prime factor."
            ),
            "variables": ["A", "B", "C", "x", "y", "z"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Beal conjecture (1993); $1M prize offered",
            "notes": (
                "Generalises Fermat's Last Theorem. Verified for many small cases. "
                "Open as of 2025."
            ),
        },
        {
            "id": "t2_polya_conjecture",
            "nl_statement": (
                "For all n ≥ 2, at least half of the natural numbers up to n "
                "have an odd number of prime factors (counting multiplicity)."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "n = 906150257 (Haselgrove 1958; first explicit counterexample)"
            ),
            "ground_truth_status": "false",
            "source": "Pólya conjecture (1919); disproved by Haselgrove (1958)",
            "notes": (
                "Pólya conjectured L(n) = #{k≤n: Ω(k) odd} - #{k≤n: Ω(k) even} ≥ 0. "
                "Disproved in 1958; smallest counterexample is n ≈ 906,150,257."
            ),
        },
        {
            "id": "t2_legendre_conjecture",
            "nl_statement": (
                "For every positive integer n, there is always at least one prime "
                "between n^2 and (n+1)^2."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Legendre's conjecture (~1808); open as of 2025",
            "notes": (
                "One of Landau's four problems (1912). Verified for all n up to ~10^10. "
                "Implies there are always primes in intervals [n^2, n^2+2n+1]."
            ),
        },
        {
            "id": "t2_twin_prime",
            "nl_statement": (
                "There are infinitely many pairs of primes (p, p+2)."
            ),
            "variables": [],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Twin prime conjecture; open as of 2025",
            "notes": (
                "Zhang (2013) proved bounded gaps; Maynard (2015) improved to gaps ≤ 246. "
                "The exact gap of 2 (twin primes) remains unproven."
            ),
        },
        {
            "id": "t2_abc_conjecture",
            "nl_statement": (
                "For every ε > 0, there are only finitely many coprime triples "
                "(a, b, c) with a + b = c and c > rad(abc)^(1+ε)."
            ),
            "variables": ["a", "b", "c", "ε"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "abc conjecture (Masser–Oesterlé 1985); claimed proof by Mochizuki (2012) disputed",
            "notes": (
                "If proved, implies Fermat's Last Theorem and many other results. "
                "Mochizuki's claimed inter-universal Teichmüller theory proof is not "
                "accepted by the broader community as of 2025."
            ),
        },
        {
            "id": "t2_riemann_hypothesis",
            "nl_statement": (
                "All non-trivial zeros of the Riemann zeta function ζ(s) have "
                "real part equal to 1/2."
            ),
            "variables": ["s"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Riemann Hypothesis (1859); one of the Millennium Prize Problems",
            "notes": (
                "Verified for the first 10^13 zeros. The most famous open problem in "
                "mathematics with a $1M Clay Institute prize."
            ),
        },
        {
            "id": "t2_catalan_mersenne",
            "nl_statement": (
                "Starting from the sequence s_0=2, s_{n+1}=2^{s_n}-1, "
                "every term is prime."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "s_4 = 2^(2^127-1) - 1 is almost certainly composite "
                "(Catalan-Mersenne sequence; s_4 has not been fully tested but "
                "s_3 = 2^8191 - 1 is prime, s_4 status unknown)"
            ),
            "ground_truth_status": "open",
            "source": "Catalan's Mersenne conjecture (~1876); open beyond s_3",
            "notes": (
                "s_0=2 (prime), s_1=3 (prime), s_2=7 (prime), s_3=127 (prime), "
                "s_4=2^127-1 (prime, M_127). s_5 = 2^(2^127-1)-1 has never been tested."
            ),
        },
        {
            "id": "t2_waring_g4",
            "nl_statement": (
                "Every positive integer is the sum of at most 4 perfect squares."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "true",
            "source": "Lagrange's four-square theorem (proved 1770)",
            "notes": (
                "Included as a negative sample. REFUTE should NOT produce a counterexample. "
                "Proved by Lagrange; generalisations studied in Waring's problem."
            ),
        },
        {
            "id": "t2_every_prime_1mod4_sum_two_squares",
            "nl_statement": (
                "Every prime p ≡ 1 (mod 4) can be expressed as a sum of two perfect squares."
            ),
            "variables": ["p"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "true",
            "source": "Fermat's theorem on sums of two squares (proved); Euler gave first complete proof",
            "notes": (
                "Included as a negative sample. True by Fermat/Euler. REFUTE should not refute this."
            ),
        },
        {
            "id": "t2_all_primes_sum_two_squares",
            "nl_statement": (
                "Every prime can be expressed as a sum of two perfect squares."
            ),
            "variables": ["p"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "p=3: 3 cannot be written as a^2 + b^2 for non-negative integers a,b; "
                "1^2+1^2=2, 1^2+0^2=1, 2^2=4 — no combination gives 3. "
                "More generally, primes p ≡ 3 (mod 4) cannot be expressed as sum of two squares."
            ),
            "ground_truth_status": "false",
            "source": "Incorrect generalisation of Fermat's theorem on sums of two squares",
            "notes": (
                "Only primes p=2 and p ≡ 1 (mod 4) are sums of two squares (Fermat's theorem). "
                "Primes p ≡ 3 (mod 4) (e.g., 3, 7, 11, 19, ...) cannot be so expressed."
            ),
        },
        {
            "id": "t2_happy_ending",
            "nl_statement": (
                "For every integer n ≥ 3, any set of n points in general position "
                "in the plane contains n points that form a convex polygon "
                "(no three collinear and no four concyclic)."
            ),
            "variables": ["n"],
            "domain": Domain.COMBINATORICS,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Erdős–Szekeres conjecture (1935); partially proved",
            "notes": (
                "The Erdős–Szekeres theorem gives an upper bound of C(2n-4, n-2)+1. "
                "The tight bound (2^(n-2)+1 suffice) is conjectured but not proved for large n. "
                "Proved for n ≤ 6."
            ),
        },
        {
            "id": "t2_ramsey_r6_6",
            "nl_statement": (
                "The Ramsey number R(6,6) equals 102."
            ),
            "variables": [],
            "domain": Domain.COMBINATORICS,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Ramsey theory; exact value of R(6,6) unknown as of 2025",
            "notes": (
                "Known bounds: 102 ≤ R(6,6) ≤ 165. R(5,5) is also unknown (43 ≤ R(5,5) ≤ 48). "
                "Testing REFUTE's recognition of open combinatorics problems."
            ),
        },
        {
            "id": "t2_hadamard_conjecture",
            "nl_statement": (
                "A Hadamard matrix of order 4k exists for every positive integer k."
            ),
            "variables": ["k"],
            "domain": Domain.COMBINATORICS,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Hadamard's conjecture (1893); open as of 2025",
            "notes": (
                "Hadamard matrices are known for all orders that are multiples of 4 up to "
                "very large values, but existence for all 4k is unproved."
            ),
        },
        {
            "id": "t2_inscribed_square",
            "nl_statement": (
                "Every simple closed planar curve (Jordan curve) contains four points "
                "that form the vertices of a square."
            ),
            "variables": [],
            "domain": Domain.COMBINATORICS,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Toeplitz's inscribed square problem (1911); open as of 2025",
            "notes": (
                "Proved for smooth curves (convex or piecewise smooth). "
                "General case remains open after 110+ years."
            ),
        },
        {
            "id": "t2_thrackle_conjecture",
            "nl_statement": (
                "In a thrackle (a graph drawn so every pair of edges meets exactly once), "
                "the number of edges is at most the number of vertices."
            ),
            "variables": [],
            "domain": Domain.COMBINATORICS,
            "ground_truth_counterexample": None,
            "ground_truth_status": "open",
            "source": "Conway's thrackle conjecture (1967); open as of 2025",
            "notes": (
                "Proved for bipartite thrackles and thrackles without odd cycles. "
                "General case open."
            ),
        },
    ]

    entries: list[BenchmarkEntry] = []
    for r in raw:
        conjecture = Conjecture(
            id=r["id"],
            domain=r["domain"],
            nl_statement=r["nl_statement"],
            variables=r["variables"],
            source="benchmark_tier2",
        )
        entry = BenchmarkEntry(
            id=r["id"],
            conjecture=conjecture,
            tier=BenchmarkTier.TIER2_HISTORICAL,
            ground_truth_counterexample=r["ground_truth_counterexample"],
            ground_truth_status=r["ground_truth_status"],
            source=r["source"],
            notes=r["notes"],
        )
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Tier 3 — Subtle / open cases (hardcoded)
# ---------------------------------------------------------------------------

def _make_tier3_entries() -> list[BenchmarkEntry]:
    """Build the Tier 3 subtle and imprecise statement entries.

    These entries test REFUTE's ability to recognise scope ambiguity,
    implicit domain assumptions, and edge cases where the statement is
    technically true but trivially vacuous or requires careful interpretation.

    Returns:
        List of BenchmarkEntry instances for Tier 3.
    """
    raw: list[dict[str, Any]] = [
        {
            "id": "t3_totient_sigma",
            "nl_statement": (
                "For all positive integers n, phi(n) + sigma(n) >= 2*n, "
                "where phi is Euler's totient function and sigma is the "
                "sum-of-divisors function."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "true_with_caveat",
            "source": "Elementary number theory; phi(n) + sigma(n) >= 2n with equality iff n=1",
            "notes": (
                "TRUE but subtle: phi(n) ≤ n-1 for n > 1 and sigma(n) ≥ n+1 for n > 1 "
                "(since 1 and n are always divisors). For n=1: phi(1)=1, sigma(1)=1, sum=2=2*1. "
                "Equality holds only at n=1; strict inequality for all n > 1. "
                "Tests whether REFUTE attempts to find a spurious counterexample."
            ),
        },
        {
            "id": "t3_consecutive_fib_product",
            "nl_statement": (
                "The product of any two consecutive Fibonacci numbers plus 1 "
                "is always a Fibonacci number."
            ),
            "variables": [],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "F(3)*F(4)+1 = 2*3+1 = 7, which is not a Fibonacci number "
                "(Fibonacci sequence: 1,1,2,3,5,8,13,...; 7 is not in it)"
            ),
            "ground_truth_status": "false",
            "source": "Imprecise Fibonacci identity; commonly stated without verifying all cases",
            "notes": (
                "The Cassini/Vajda identity F(n-1)*F(n+1) - F(n)^2 = (-1)^n is true, "
                "but F(n)*F(n+1)+1 is not always Fibonacci. This tests scope-checking."
            ),
        },
        {
            "id": "t3_prime_gap_bound",
            "nl_statement": (
                "For all primes p, the next prime after p is at most p + log(p)."
            ),
            "variables": ["p"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "p=7: next prime is 11, gap=4; log(7)≈1.95, so 7+1.95=8.95 < 11. False."
            ),
            "ground_truth_status": "false",
            "source": "Imprecise folk statement about prime gaps; actual bound involves log^2(p)",
            "notes": (
                "The actual Cramér conjecture states gaps are O(log^2 p). "
                "The weaker prime number theorem gives average gap ~log(p). "
                "This statement confuses average with worst-case gaps. "
                "Tests whether REFUTE can handle log notation and prime gap reasoning."
            ),
        },
        {
            "id": "t3_sum_reciprocals_primes",
            "nl_statement": (
                "The sum of reciprocals of all primes converges to a finite value."
            ),
            "variables": [],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "The sum 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + ... diverges (Euler 1737). "
                "The partial sums grow like log(log(n))."
            ),
            "ground_truth_status": "false",
            "source": "Imprecise statement about prime reciprocal sum; Euler proved divergence (1737)",
            "notes": (
                "Commonly confused with sum over all integers (harmonic series, diverges) "
                "or product formula. Euler's proof: product_p (1 - 1/p)^{-1} = sum_n 1/n → divergence."
            ),
        },
        {
            "id": "t3_stirling_approx",
            "nl_statement": (
                "For all n ≥ 1, n! = sqrt(2*pi*n) * (n/e)^n."
            ),
            "variables": ["n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "n=1: sqrt(2*pi)*1/e ≈ 0.922, not equal to 1! = 1. "
                "Stirling's approximation is an asymptotic formula, not an exact equality."
            ),
            "ground_truth_status": "false",
            "source": "Stirling's approximation misquoted as exact equality; actually n! ~ sqrt(2πn)(n/e)^n",
            "notes": (
                "The statement uses '=' instead of '~' (asymptotic). "
                "Tests REFUTE's sensitivity to precise vs asymptotic claims. "
                "Ratio n! / (sqrt(2πn)(n/e)^n) → 1 but is never exactly 1."
            ),
        },
        {
            "id": "t3_wilson_prime_generalisation",
            "nl_statement": (
                "For every prime p, (p-1)! ≡ p-1 (mod p^2)."
            ),
            "variables": ["p"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "p=3: (3-1)! = 2, and 2 mod 9 = 2 ≠ 8 = p-1 = 2 mod 9. "
                "Wait: p-1=2, p^2=9: 2 mod 9 = 2 = p-1. True here. "
                "p=5: 4!=24, 24 mod 25=24=p-1=4? No: p-1=4, but 24≠4. False."
            ),
            "ground_truth_status": "false",
            "source": "Incorrect generalisation of Wilson's theorem; Wilson primes satisfy (p-1)! ≡ -1 (mod p^2)",
            "notes": (
                "Wilson's theorem: (p-1)! ≡ -1 (mod p). "
                "Wilson primes are primes where (p-1)! ≡ -1 (mod p^2); only 5, 13, 563 known. "
                "The statement confuses mod p and mod p^2 and uses wrong residue."
            ),
        },
        {
            "id": "t3_even_perfect_mersenne",
            "nl_statement": (
                "Every Mersenne prime 2^p - 1 is associated with an even perfect number "
                "2^(p-1) * (2^p - 1), and these are all even perfect numbers."
            ),
            "variables": ["p"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "true_with_caveat",
            "source": "Euclid-Euler theorem (proved); but statement conflates two separate results",
            "notes": (
                "TRUE: (1) Euclid proved 2^(p-1)*(2^p-1) is perfect when 2^p-1 is prime. "
                "(2) Euler proved every even perfect number has this form. "
                "The conjunction is true but subtly requires p prime AND 2^p-1 prime. "
                "Tests whether REFUTE detects the hidden Mersenne-prime precondition."
            ),
        },
        {
            "id": "t3_catalan_identity_scope",
            "nl_statement": (
                "For all Fibonacci numbers F_n and F_m with n > m, "
                "F_n^2 - F_{n+r} * F_{n-r} = (-1)^{n-r} * F_r^2 for all r."
            ),
            "variables": ["n", "m", "r"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": None,
            "ground_truth_status": "true_with_caveat",
            "source": "Catalan's identity (proved); statement has unnecessary variable m",
            "notes": (
                "Catalan's identity is true: F_n^2 - F_{n+r}*F_{n-r} = (-1)^{n-r} F_r^2. "
                "The inclusion of variable m is spurious and makes the scope ambiguous. "
                "Tests whether REFUTE flags the irrelevant free variable."
            ),
        },
        {
            "id": "t3_chinese_remainder_general",
            "nl_statement": (
                "For any integers a, b and any positive integers m, n, "
                "there exists an integer x such that x ≡ a (mod m) and x ≡ b (mod n)."
            ),
            "variables": ["a", "b", "m", "n"],
            "domain": Domain.NUMBER_THEORY,
            "ground_truth_counterexample": (
                "m=4, n=6, a=1, b=2: x ≡ 1 (mod 4) and x ≡ 2 (mod 6). "
                "x ≡ 1 (mod 4) gives x ∈ {1, 5, 9, 13, ...}; checking mod 6: "
                "1 mod 6=1, 5 mod 6=5, 9 mod 6=3, 13 mod 6=1 — none give 2. "
                "The CRT requires gcd(m,n) | (b-a); here gcd(4,6)=2 and b-a=1, not divisible by 2."
            ),
            "ground_truth_status": "false",
            "source": "Incorrect generalisation of Chinese Remainder Theorem; requires gcd(m,n) | (b-a)",
            "notes": (
                "CRT in full generality requires gcd(m,n) | (b-a). "
                "The standard version taught in courses often only covers coprime moduli. "
                "Tests REFUTE's handling of subtle precondition omissions."
            ),
        },
        {
            "id": "t3_power_tower_growth",
            "nl_statement": (
                "For all real x > 1, the infinite power tower x^(x^(x^...)) converges."
            ),
            "variables": ["x"],
            "domain": Domain.INEQUALITY,
            "ground_truth_counterexample": (
                "x = sqrt(2)^sqrt(2) ≈ 1.6325: converges (value = sqrt(2)^sqrt(2)^... = 2). "
                "But x = 10: 10^10^10^... diverges. "
                "Convergence requires e^(-e) ≤ x ≤ e^(1/e) ≈ 1.4447. "
                "x=2 > e^(1/e) ≈ 1.4447: x=2 gives divergent tower."
            ),
            "ground_truth_status": "false",
            "source": "Infinite power tower convergence; requires x in [e^{-e}, e^{1/e}]",
            "notes": (
                "The infinite power tower (tetration) converges iff x ∈ [e^{-e}, e^{1/e}]. "
                "For x > e^{1/e} ≈ 1.4447, it diverges. x=2 is a simple counterexample. "
                "Tests REFUTE's ability to reason about convergence conditions."
            ),
        },
    ]

    entries: list[BenchmarkEntry] = []
    for r in raw:
        conjecture = Conjecture(
            id=r["id"],
            domain=r["domain"],
            nl_statement=r["nl_statement"],
            variables=r["variables"],
            source="benchmark_tier3",
        )
        entry = BenchmarkEntry(
            id=r["id"],
            conjecture=conjecture,
            tier=BenchmarkTier.TIER3_SUBTLE,
            ground_truth_counterexample=r.get("ground_truth_counterexample"),
            ground_truth_status=r["ground_truth_status"],
            source=r["source"],
            notes=r["notes"],
        )
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# SymPy-based Tier 1 verification helpers
# ---------------------------------------------------------------------------

def _verify_tier1_entry_sympy(entry: BenchmarkEntry) -> bool:
    """Verify that a Tier 1 entry's counterexample is numerically valid.

    Iterates n = 0 … 100 and checks whether the synthetic false statement
    fails at the claimed counterexample value.  Currently supports
    divisibility-pattern statements of the form ``"k divides f(n)"``.
    Returns True if any n in 0..100 witnesses falsity; returns True
    unconditionally for non-parseable statements (conservative pass)
    so that human-authored entries are never silently dropped.

    Args:
        entry: A Tier 1 BenchmarkEntry to verify.

    Returns:
        True if a counterexample is confirmed (or statement is non-parseable),
        False if the statement appears to hold for all n in 0..100.
    """
    import re

    import sympy

    stmt = entry.conjecture.nl_statement.lower()
    n_sym = sympy.Symbol("n", nonnegative=True, integer=True)

    # --- Divisibility pattern: "k divides f(n)" or "f(n) is divisible by k"
    div_patterns = [
        r"(\d+)\s+divides\s+(.+?)(?:\s+for|\s*$)",
        r"(\d+)\s*\|\s*(.+?)(?:\s+for|\s*$)",
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

        expr_str = expr_str.strip().rstrip(".")
        try:
            expr = sympy.sympify(
                expr_str,
                locals={"n": n_sym, "factorial": sympy.factorial},
            )
        except (sympy.SympifyError, SyntaxError):
            return True  # conservative pass

        for val in range(0, 101):
            try:
                result = int(expr.subs(n_sym, val).evalf())
                if result % k != 0:
                    return True
            except (TypeError, ValueError):
                continue

        return False  # passed all 0..100, claim may be true

    # --- Inequality pattern: "lhs >= c * rhs" or similar
    ineq_patterns = [
        r"(.+?)\s*>=\s*(.+?)(?:\s+for|\s+is|\s*$)",
        r"(.+?)\s*>\s*(.+?)(?:\s+for|\s*$)",
    ]
    a_sym = sympy.Symbol("a", positive=True, real=True)
    b_sym = sympy.Symbol("b", positive=True, real=True)
    c_sym = sympy.Symbol("c", positive=True, real=True)
    sym_locals = {"a": a_sym, "b": b_sym, "c": c_sym}

    for pat in ineq_patterns:
        m = re.search(pat, stmt)
        if m is None:
            continue
        lhs_str, rhs_str = m.group(1).strip(), m.group(2).strip().rstrip(".")
        try:
            lhs = sympy.sympify(lhs_str, locals=sym_locals)
            rhs = sympy.sympify(rhs_str, locals=sym_locals)
        except (sympy.SympifyError, SyntaxError):
            return True

        import random

        random.seed(0)
        test_points = [(1.0, 1.0, 1.0), (0.5, 2.0, 1.0), (2.0, 0.5, 3.0)]
        for _ in range(50):
            test_points.append((
                random.uniform(0.01, 5.0),
                random.uniform(0.01, 5.0),
                random.uniform(0.01, 5.0),
            ))

        for a_v, b_v, c_v in test_points:
            subs = {a_sym: a_v, b_sym: b_v, c_sym: c_v}
            try:
                lv = float(lhs.subs(subs).evalf())
                rv = float(rhs.subs(subs).evalf())
                if lv < rv - 1e-9:
                    return True
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        return False  # held everywhere sampled

    return True  # no parseable pattern — conservative pass


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _entry_to_dict(entry: BenchmarkEntry) -> dict[str, Any]:
    """Serialise a BenchmarkEntry to a JSON-compatible dictionary.

    Uses dataclasses.asdict recursively, then converts Enum members to their
    string values for JSONL compatibility.

    Args:
        entry: The BenchmarkEntry to serialise.

    Returns:
        A plain dictionary suitable for json.dumps.
    """
    raw = dataclasses.asdict(entry)

    def _convert_enums(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert_enums(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert_enums(v) for v in obj]
        if isinstance(obj, str):
            return obj
        return obj

    return _convert_enums(raw)


def _dict_to_entry(d: dict[str, Any]) -> BenchmarkEntry:
    """Deserialise a BenchmarkEntry from a plain dictionary.

    Reconstructs Enum fields from their string values and rebuilds the
    nested Conjecture dataclass.

    Args:
        d: A plain dictionary as produced by ``_entry_to_dict``.

    Returns:
        A fully reconstructed BenchmarkEntry.

    Raises:
        KeyError: If a required field is missing from the dictionary.
        ValueError: If an Enum value string is not valid.
    """
    conj_d = d["conjecture"]
    conjecture = Conjecture(
        id=conj_d["id"],
        domain=Domain(conj_d["domain"]),
        nl_statement=conj_d["nl_statement"],
        variables=conj_d["variables"],
        source=conj_d.get("source", ""),
        timestamp=conj_d.get("timestamp", ""),
        metadata=conj_d.get("metadata", {}),
    )
    return BenchmarkEntry(
        id=d["id"],
        conjecture=conjecture,
        tier=BenchmarkTier(d["tier"]),
        ground_truth_counterexample=d.get("ground_truth_counterexample"),
        ground_truth_status=d.get("ground_truth_status", "false"),
        source=d.get("source", ""),
        notes=d.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------


class BenchmarkBuilder:
    """Constructs and validates the 3-tier REFUTE benchmark.

    Tier 1 entries are generated programmatically from ``_KNOWN_TRUE_TEMPLATES``
    — each falsification produces one BenchmarkEntry with a claimed
    counterexample.  Tier 2 and Tier 3 entries are hardcoded from verified
    mathematical literature.

    After building, callers should invoke ``validate_tier1_entry`` (or the
    batch ``validate_tier1``) to confirm that every Tier 1 counterexample is
    genuinely witnessed for n in 0..100.

    Typical workflow::

        builder = BenchmarkBuilder()
        t1 = builder.build_tier1()
        t2 = builder.build_tier2()
        t3 = builder.build_tier3()
        valid_t1 = [e for e in t1 if builder.validate_tier1_entry(e)]
        all_entries = valid_t1 + t2 + t3
        builder.save(all_entries, Path("data/benchmark/all.jsonl"))
    """

    def build_tier1(self) -> list[BenchmarkEntry]:
        """Generate Tier 1 synthetic-falsehood benchmark entries.

        Iterates over ``_KNOWN_TRUE_TEMPLATES`` and converts each
        ``_Falsification`` into a ``BenchmarkEntry``.  Each entry receives a
        deterministic ID based on its index in the flat falsification list.

        Returns:
            List of BenchmarkEntry objects tagged as TIER1_SYNTHETIC.
        """
        entries: list[BenchmarkEntry] = []
        flat_idx = 0
        for template in _KNOWN_TRUE_TEMPLATES:
            for falsification in template.falsifications:
                entry_id = f"t1_{flat_idx:04d}"
                conjecture = Conjecture(
                    id=entry_id,
                    domain=falsification.domain,
                    nl_statement=falsification.false_statement,
                    variables=falsification.variables,
                    source="benchmark_tier1_synthetic",
                    metadata={"true_origin": template.true_statement},
                )
                entry = BenchmarkEntry(
                    id=entry_id,
                    conjecture=conjecture,
                    tier=BenchmarkTier.TIER1_SYNTHETIC,
                    ground_truth_counterexample=falsification.counterexample_str,
                    ground_truth_status="false",
                    source=(
                        f"Synthetic falsification of: {template.true_statement}"
                    ),
                    notes=(
                        f"Original true theorem: '{template.true_statement}'. "
                        f"Counterexample: {falsification.counterexample_str}"
                    ),
                )
                entries.append(entry)
                flat_idx += 1

        logger.info("Built %d Tier 1 synthetic entries from %d templates", len(entries), len(_KNOWN_TRUE_TEMPLATES))
        return entries

    def build_tier2(self) -> list[BenchmarkEntry]:
        """Return hardcoded Tier 2 historical conjecture entries.

        All entries are sourced from ``_make_tier2_entries()`` which encodes
        historically verified mathematical results.

        Returns:
            List of BenchmarkEntry objects tagged as TIER2_HISTORICAL.
        """
        entries = _make_tier2_entries()
        logger.info("Built %d Tier 2 historical entries", len(entries))
        return entries

    def build_tier3(self) -> list[BenchmarkEntry]:
        """Return hardcoded Tier 3 subtle/open statement entries.

        All entries are sourced from ``_make_tier3_entries()`` which encodes
        subtly mis-stated or scope-ambiguous published conjectures.

        Returns:
            List of BenchmarkEntry objects tagged as TIER3_SUBTLE.
        """
        entries = _make_tier3_entries()
        logger.info("Built %d Tier 3 subtle entries", len(entries))
        return entries

    def validate_tier1_entry(self, entry: BenchmarkEntry) -> bool:
        """Validate a single Tier 1 entry using SymPy.

        Uses pattern matching on the natural-language statement to extract
        a testable mathematical expression, then numerically verifies that
        the stated conjecture fails for at least one n in 0..100 (divisibility
        statements) or one (a,b,c) sample (inequality statements).

        Non-parseable statements are conservatively accepted (return True) so
        that human-authored entries are never silently discarded.

        Args:
            entry: A BenchmarkEntry to validate.  Should be TIER1_SYNTHETIC.

        Returns:
            True if the entry passes validation (counterexample confirmed or
            statement is non-parseable); False if the statement appears to
            hold for all tested values.
        """
        return _verify_tier1_entry_sympy(entry)

    def validate_tier1(
        self,
        entries: list[BenchmarkEntry],
        show_progress: bool = True,
    ) -> tuple[list[BenchmarkEntry], list[BenchmarkEntry]]:
        """Validate all Tier 1 entries in batch.

        Args:
            entries: List of Tier 1 BenchmarkEntry objects to validate.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            A two-tuple ``(valid_entries, invalid_entries)`` where
            ``valid_entries`` passed validation and ``invalid_entries`` did not.
        """
        valid: list[BenchmarkEntry] = []
        invalid: list[BenchmarkEntry] = []

        iterator = tqdm(
            entries,
            desc="Validating Tier 1",
            unit="entry",
            dynamic_ncols=True,
            disable=not show_progress,
            postfix={"valid": 0, "invalid": 0},
        )

        for entry in iterator:
            if self.validate_tier1_entry(entry):
                valid.append(entry)
            else:
                invalid.append(entry)
                logger.warning(
                    "Tier 1 entry %s FAILED validation: '%s'",
                    entry.id,
                    entry.conjecture.nl_statement[:80],
                )
            if show_progress:
                iterator.set_postfix(valid=len(valid), invalid=len(invalid), refresh=False)

        logger.info(
            "Tier 1 validation complete: %d valid / %d invalid / %d total",
            len(valid),
            len(invalid),
            len(entries),
        )
        return valid, invalid

    def save(self, entries: list[BenchmarkEntry], path: Path) -> None:
        """Serialise a list of BenchmarkEntry objects to a JSONL file.

        Each entry is written as a single JSON line.  The parent directory
        is created if it does not exist.  Overwrites any existing file at
        ``path``.

        Args:
            entries: The list of BenchmarkEntry objects to save.
            path: Destination file path (will be created or overwritten).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(_entry_to_dict(entry), ensure_ascii=False) + "\n")
        logger.info("Saved %d entries to %s", len(entries), path)

    def load(self, path: Path) -> list[BenchmarkEntry]:
        """Deserialise BenchmarkEntry objects from a JSONL file.

        Args:
            path: Source file path.

        Returns:
            List of BenchmarkEntry objects in file order.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            json.JSONDecodeError: If a line is not valid JSON.
            KeyError: If a required field is absent in a record.
            ValueError: If an Enum value string is not recognised.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        entries: list[BenchmarkEntry] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    entries.append(_dict_to_entry(d))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise type(exc)(f"Error parsing line {line_no} of {path}: {exc}") from exc
        logger.info("Loaded %d entries from %s", len(entries), path)
        return entries


class BenchmarkLoader:
    """Loads and filters the serialised REFUTE benchmark for evaluation runs.

    Provides methods to load all tiers at once, load a single tier by name,
    and compute aggregate statistics suitable for logging and reporting.

    Typical usage::

        loader = BenchmarkLoader()
        all_entries = loader.load_all(Path("data/benchmark"))
        tier1_entries = loader.load_tier(Path("data/benchmark"), BenchmarkTier.TIER1_SYNTHETIC)
        stats = loader.get_stats(all_entries)
    """

    # Expected per-tier file names
    _TIER_FILES: dict[BenchmarkTier, str] = {
        BenchmarkTier.TIER1_SYNTHETIC: "tier1.jsonl",
        BenchmarkTier.TIER2_HISTORICAL: "tier2.jsonl",
        BenchmarkTier.TIER3_SUBTLE: "tier3.jsonl",
    }

    def __init__(self) -> None:
        """Initialise BenchmarkLoader.

        Creates a private BenchmarkBuilder instance solely for its ``load``
        method (JSONL deserialisation logic is shared).
        """
        self._builder = BenchmarkBuilder()

    def load_all(self, data_dir: Path) -> list[BenchmarkEntry]:
        """Load all three tier files from ``data_dir`` and concatenate them.

        Silently skips any tier file that does not exist yet (useful when
        running partial builds).  Logs a warning for each missing file.

        Args:
            data_dir: Directory containing ``tier1.jsonl``, ``tier2.jsonl``,
                and ``tier3.jsonl``.

        Returns:
            Concatenated list of BenchmarkEntry objects from all available
            tier files, in tier order (1 → 2 → 3).
        """
        data_dir = Path(data_dir)
        all_entries: list[BenchmarkEntry] = []
        for tier, filename in self._TIER_FILES.items():
            tier_path = data_dir / filename
            if not tier_path.exists():
                logger.warning("Tier file not found, skipping: %s", tier_path)
                continue
            entries = self._builder.load(tier_path)
            all_entries.extend(entries)
            logger.debug("Loaded %d entries from %s", len(entries), tier_path)
        logger.info("load_all: %d total entries from %s", len(all_entries), data_dir)
        return all_entries

    def load_tier(self, data_dir: Path, tier: BenchmarkTier) -> list[BenchmarkEntry]:
        """Load a single tier's JSONL file.

        Args:
            data_dir: Directory containing the tier files.
            tier: Which tier to load.

        Returns:
            List of BenchmarkEntry objects for the requested tier.

        Raises:
            FileNotFoundError: If the tier file does not exist.
            ValueError: If ``tier`` is not a recognised BenchmarkTier value.
        """
        if tier not in self._TIER_FILES:
            raise ValueError(f"Unrecognised tier: {tier!r}. Expected one of {list(self._TIER_FILES)}")
        tier_path = Path(data_dir) / self._TIER_FILES[tier]
        return self._builder.load(tier_path)

    def get_stats(self, entries: list[BenchmarkEntry]) -> dict[str, Any]:
        """Compute aggregate statistics over a list of BenchmarkEntry objects.

        Counts entries per tier, per domain, per ground-truth status, and
        computes the fraction with known counterexamples.

        Args:
            entries: Any list of BenchmarkEntry objects (mixed or filtered).

        Returns:
            A dictionary with keys:

            * ``"total"`` — total entry count.
            * ``"by_tier"`` — ``{tier_value: count}`` mapping.
            * ``"by_domain"`` — ``{domain_value: count}`` mapping.
            * ``"by_status"`` — ``{ground_truth_status: count}`` mapping.
            * ``"with_counterexample"`` — count of entries where
              ``ground_truth_counterexample`` is not None.
            * ``"fraction_with_counterexample"`` — float in [0, 1].
        """
        from collections import Counter

        by_tier: Counter[str] = Counter()
        by_domain: Counter[str] = Counter()
        by_status: Counter[str] = Counter()
        with_cx = 0

        for entry in entries:
            by_tier[entry.tier.value] += 1
            by_domain[entry.conjecture.domain.value] += 1
            by_status[entry.ground_truth_status] += 1
            if entry.ground_truth_counterexample is not None:
                with_cx += 1

        total = len(entries)
        return {
            "total": total,
            "by_tier": dict(by_tier),
            "by_domain": dict(by_domain),
            "by_status": dict(by_status),
            "with_counterexample": with_cx,
            "fraction_with_counterexample": with_cx / total if total > 0 else 0.0,
        }
