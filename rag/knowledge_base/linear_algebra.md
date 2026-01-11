---
topic: linear_algebra
type: formula
---

# Linear Algebra Basics

## Matrix Operations
- (AB)ᵀ = BᵀAᵀ
- (A⁻¹)ᵀ = (Aᵀ)⁻¹
- det(AB) = det(A) · det(B)
- det(A⁻¹) = 1/det(A)

## 2x2 Matrix Inverse
If A = [a b; c d], then:
A⁻¹ = (1/det(A)) · [d -b; -c a]
det(A) = ad - bc

## Eigenvalues
For Ax = λx:
- det(A - λI) = 0 gives eigenvalues
- Sum of eigenvalues = trace(A)
- Product of eigenvalues = det(A)

## Systems of Linear Equations
Cramer's Rule: xᵢ = det(Aᵢ)/det(A)

---
topic: linear_algebra
type: common_mistakes
---

# Common Mistakes in Linear Algebra

1. **Matrix multiplication**: AB ≠ BA in general (not commutative)
2. **Inverse**: Not all matrices have inverses (det = 0)
3. **Determinant of sum**: det(A+B) ≠ det(A) + det(B)
4. **Row operations**: Forgetting effect on determinant
5. **Eigenvalues**: Confusing algebraic and geometric multiplicity