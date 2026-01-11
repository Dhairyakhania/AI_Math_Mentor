---
topic: probability
type: formula
---

# Probability Formulas

## Basic Probability
- P(A) = favorable outcomes / total outcomes
- 0 ≤ P(A) ≤ 1
- P(A') = 1 - P(A)

## Addition Rules
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- If A, B mutually exclusive: P(A ∪ B) = P(A) + P(B)

## Multiplication Rules
- P(A ∩ B) = P(A) · P(B|A)
- If A, B independent: P(A ∩ B) = P(A) · P(B)

## Conditional Probability
P(A|B) = P(A ∩ B) / P(B)

## Bayes' Theorem
P(A|B) = P(B|A) · P(A) / P(B)

## Permutations and Combinations
- nPr = n! / (n-r)!
- nCr = n! / (r!(n-r)!)

## Binomial Distribution
P(X = k) = nCk · pᵏ · (1-p)ⁿ⁻ᵏ

---
topic: probability  
type: common_mistakes
---

# Common Mistakes in Probability

1. **Independence vs Mutual Exclusivity**: These are different concepts!
2. **Conditional probability**: Confusing P(A|B) with P(B|A)
3. **Counting**: Not accounting for order when it matters
4. **Complementary events**: Forgetting P(A') = 1 - P(A)
5. **Bayes' theorem**: Incorrectly identifying prior and posterior