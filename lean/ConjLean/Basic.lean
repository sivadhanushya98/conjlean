/-
  ConjLean Example Library
  ========================

  This file serves as the canonical style reference for theorem statements
  and proofs produced by the ConjLean pipeline.  It contains hand-written
  examples that demonstrate the target proof style: short automation-friendly
  proofs using Mathlib4 tactics such as `omega`, `simp`, `positivity`, and
  `nlinarith`.

  All theorems here are fully proved — no `sorry` is used anywhere in this file.

  The automated pipeline writes theorems in the same format: a Lean 4 `theorem`
  declaration whose body is discharged by a tactic proof.
-/

import Mathlib

namespace ConjLean

/-!
## Example 1: Divisibility of Consecutive Integers

Every product of three consecutive natural numbers is divisible by 3.

Proof strategy: induction on `n`, with the inductive step closed by `omega`
after ring normalization.
-/

theorem example_div_3 (n : ℕ) : 3 ∣ n * (n + 1) * (n + 2) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have : n * (n + 1) * (n + 2) + 3 * (n + 1) * (n + 2) =
           (n + 1) * (n + 2) * (n + 3) := by ring
    have hdvd : 3 ∣ 3 * (n + 1) * (n + 2) := dvd_mul_right 3 _
    linarith [Nat.dvd_add ih hdvd]

/-!
## Example 2: Squares are Non-Negative over the Integers

For any integer `a`, we have `0 ≤ a ^ 2`.

Proof strategy: `nlinarith` with the auxiliary fact `a ^ 2 = a * a` is
sufficient; alternatively `positivity` works directly.
-/

theorem example_sq_nonneg (a : ℤ) : 0 ≤ a ^ 2 := by
  positivity

/-!
## Example 3: GCD of a Number with Itself

For any natural number `n`, `Nat.gcd n n = n`.

Proof strategy: `simp` discharges the goal using the `Nat.gcd_self` simp lemma
that is part of Mathlib.
-/

theorem example_gcd_self (n : ℕ) : Nat.gcd n n = n := by
  simp [Nat.gcd_self]

/-!
## Bonus: Parity of n² + n

For all natural numbers `n`, the expression `n ^ 2 + n` is even.

Proof strategy: the goal follows by `omega` after observing that
`n ^ 2 + n = n * (n + 1)`, and exactly one of `n`, `n + 1` is even.
-/

theorem example_sq_add_n_even (n : ℕ) : 2 ∣ n ^ 2 + n := by
  have : n ^ 2 + n = n * (n + 1) := by ring
  rw [this]
  rcases Nat.even_or_odd n with ⟨k, hk⟩ | ⟨k, hk⟩
  · exact ⟨k * (2 * k + 1), by omega⟩
  · exact ⟨(2 * k + 1) * (k + 1), by omega⟩

end ConjLean
