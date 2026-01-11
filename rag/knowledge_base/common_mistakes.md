---
topic: general
type: common_mistakes
---

# General Mathematical Mistakes

## Algebraic Errors

### Sign Errors
- Forgetting to distribute negative signs: -(a + b) = -a - b, NOT -a + b
- Losing track of signs when moving terms across equals sign
- Double negative confusion: -(-x) = x

### Order of Operations (PEMDAS/BODMAS)
- Evaluating left to right without respecting precedence
- Forgetting that exponents come before multiplication
- Not handling nested parentheses correctly

### Fraction Mistakes
- Adding numerators without common denominator: a/b + c/d ≠ (a+c)/(b+d)
- Incorrectly simplifying: (a+b)/c ≠ a/c + b/c is CORRECT, but a/(b+c) ≠ a/b + a/c
- Forgetting to flip when dividing by fraction

---
topic: algebra
type: common_mistakes
---

# Algebra Common Mistakes

## Quadratic Equations
1. **Missing ± in quadratic formula**: x = (-b ± √(b²-4ac))/2a - both roots needed!
2. **Sign error in discriminant**: b²-4ac, not b²+4ac
3. **Division error**: Dividing by 2a, not just 2
4. **Forgetting to check**: Always verify roots by substitution

## Factoring
1. **Incomplete factoring**: x² - 4 = (x-2)(x+2), not just (x-2)²
2. **Missing GCF**: Always check for greatest common factor first
3. **Sign errors in trinomials**: x² - 5x + 6 = (x-2)(x-3), check signs carefully

## Exponents and Logarithms
1. **Adding exponents incorrectly**: a^m × a^n = a^(m+n), NOT a^(mn)
2. **Power of power**: (a^m)^n = a^(mn), NOT a^(m+n)
3. **Log of sum**: log(a+b) ≠ log(a) + log(b)
4. **Log of product**: log(ab) = log(a) + log(b) - this IS correct

## Inequalities
1. **Forgetting to flip sign**: When multiplying/dividing by negative, flip the inequality
2. **Treating like equation**: |x| < 3 means -3 < x < 3, not x < 3 or x < -3
3. **Interval notation**: Check open vs closed brackets

---
topic: calculus
type: common_mistakes
---

# Calculus Common Mistakes

## Limits
1. **Direct substitution when indeterminate**: 0/0 requires further analysis
2. **L'Hôpital's misuse**: Only for 0/0 or ∞/∞ forms
3. **One-sided limits**: Check both sides at discontinuities
4. **Infinity arithmetic**: ∞ - ∞ is indeterminate, not 0

## Derivatives
1. **Chain rule forgotten**: d/dx[f(g(x))] = f'(g(x)) · g'(x), NOT just f'(g(x))
2. **Product rule error**: (fg)' = f'g + fg', NOT f'g'
3. **Quotient rule sign**: (f/g)' = (f'g - fg')/g², note the minus sign
4. **Implicit differentiation**: Don't forget dy/dx when differentiating y

## Common Derivative Errors
- d/dx(sin x) = cos x, NOT -cos x
- d/dx(cos x) = -sin x, NOT sin x
- d/dx(tan x) = sec²x, NOT sec x tan x
- d/dx(e^x) = e^x, NOT x·e^(x-1)
- d/dx(ln x) = 1/x, NOT ln x / x

## Integration
1. **Forgetting +C**: Indefinite integrals need constant of integration
2. **u-substitution**: Don't forget to change limits in definite integrals
3. **Integration by parts**: Choose u and dv wisely (LIATE rule)
4. **Absolute value**: ∫1/x dx = ln|x| + C, need absolute value

---
topic: probability
type: common_mistakes
---

# Probability Common Mistakes

## Basic Probability
1. **Probability > 1**: Probability must be between 0 and 1
2. **Complementary events**: P(A') = 1 - P(A), not P(A) - 1
3. **Sample space errors**: Make sure to count all outcomes correctly

## Conditional Probability
1. **Confusing P(A|B) and P(B|A)**: These are generally different!
2. **Bayes' theorem application**: Identify prior and posterior correctly
3. **Independence assumption**: Don't assume independence without justification

## Counting
1. **Permutation vs Combination**: Order matters in permutations, not in combinations
2. **Overcounting**: Watch for duplicate arrangements
3. **With/without replacement**: Affects probability calculations significantly

## Common Formulas Misapplied
1. **Addition rule**: P(A∪B) = P(A) + P(B) - P(A∩B), don't forget to subtract intersection
2. **Multiplication for AND**: P(A∩B) = P(A)·P(B|A), not always P(A)·P(B)
3. **Mutually exclusive vs Independent**: These are different concepts!

---
topic: linear_algebra
type: common_mistakes
---

# Linear Algebra Common Mistakes

## Matrix Operations
1. **Non-commutativity**: AB ≠ BA in general
2. **Scalar multiplication**: k(A+B) = kA + kB, but (A+B)² ≠ A² + 2AB + B²
3. **Matrix multiplication dimensions**: (m×n)(n×p) = (m×p), dimensions must match

## Determinants
1. **Determinant of sum**: det(A+B) ≠ det(A) + det(B)
2. **Row operations effect**: Row swap changes sign, row multiplication scales determinant
3. **Zero determinant**: Means matrix is singular (no inverse)

## Inverse Matrices
1. **Not all matrices invertible**: Check det(A) ≠ 0 first
2. **Inverse of product**: (AB)⁻¹ = B⁻¹A⁻¹, note the reversed order
3. **2×2 formula**: A⁻¹ = (1/det(A))[d -b; -c a], watch the signs

## Systems of Equations
1. **Infinite solutions**: Dependent equations need parameter
2. **No solution**: Inconsistent system, check augmented matrix
3. **Gaussian elimination errors**: Keep track of row operations carefully

## Eigenvalues/Eigenvectors
1. **Characteristic equation**: det(A - λI) = 0, not det(A - λ)
2. **Eigenvector normalization**: Remember eigenvectors aren't unique (scalar multiples)
3. **Algebraic vs geometric multiplicity**: May differ for defective matrices

---
topic: trigonometry
type: common_mistakes
---

# Trigonometry Common Mistakes

## Basic Identities
1. **Pythagorean identity**: sin²x + cos²x = 1, NOT sin²x - cos²x = 1
2. **Reciprocal confusion**: sec x = 1/cos x, NOT 1/sin x
3. **Negative angles**: sin(-x) = -sin(x), cos(-x) = cos(x)

## Angle Measurement
1. **Degrees vs Radians**: π radians = 180°, always check calculator mode
2. **Reference angles**: Find correct quadrant for general angles
3. **Period awareness**: sin and cos have period 2π, tan has period π

## Common Values
- sin(0) = 0, sin(π/6) = 1/2, sin(π/4) = √2/2, sin(π/3) = √3/2, sin(π/2) = 1
- cos values are reverse order
- tan(π/4) = 1, tan(0) = 0, tan(π/2) is undefined

## Equations
1. **General solutions**: Don't forget to add 2nπ or nπ for all solutions
2. **Squaring both sides**: May introduce extraneous solutions, always verify
3. **Domain restrictions**: tan x undefined at π/2 + nπ