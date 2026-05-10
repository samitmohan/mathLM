"""Generate synthetic ML/DL math Q&A pairs → ml_math.bin.

Covers linear algebra (matrix gradients, eigenvalues, SVD), probability &
statistics (Bayes, MLE, KL divergence), loss functions, backprop, attention,
optimisers, and information theory.

    python -m mathlm.data.generate_ml_math
    python -m mathlm.data.generate_ml_math --check
"""

import sys
import math
import random
import itertools
import numpy as np

from mathlm.model.tokenizer import MathTokenizer as _MathTok
if _MathTok.is_available():
    _tok = _MathTok(); _tok.load()
    tokenise = _tok.encode
else:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    tokenise = lambda text: _enc.encode(text, disallowed_special=())


def _qa(q: str, a: str) -> str:
    return f"Q: {q}\nA: {a}\n\n"


def fmt_float(x, d=4):
    return f"{x:.{d}f}".rstrip('0').rstrip('.')

def gen_matrix_gradient(rng):
    pairs = []

    # ∇_x (x^T A x) for symmetric A — conceptual variants
    templates_xTAx = [
        ("What is the gradient of f(x) = x^T A x with respect to x, for symmetric A?",
         "∇f(x) = 2Ax. For symmetric A, the gradient of the quadratic form x^T A x is 2Ax."),
        ("Compute ∇_x (x^T A x) when A is not necessarily symmetric.",
         "∇_x (x^T A x) = (A + A^T)x. When A is symmetric this becomes 2Ax."),
        ("If f(w) = w^T A w and A is positive definite symmetric, what is ∇f(w)?",
         "∇f(w) = 2Aw. The gradient of a quadratic form with symmetric A is 2Aw."),
        ("Differentiate f(x) = (1/2) x^T Q x + b^T x + c with respect to x.",
         "∇f(x) = (1/2)(Q + Q^T)x + b. If Q is symmetric: ∇f(x) = Qx + b."),
    ]
    pairs.extend([_qa(q, a) for q, a in templates_xTAx])

    # ∇_x (a^T x) variants
    for name in ["a", "c", "v", "u"]:
        pairs.append(_qa(
            f"What is the gradient of f(x) = {name}^T x with respect to x?",
            f"∇_x ({name}^T x) = {name}. The gradient of a linear function {name}^T x is the coefficient vector {name}."
        ))

    # ||x - b||^2 gradient with various variable names
    for (x, b) in [("x","b"), ("w","w*"), ("θ","μ"), ("z","z0")]:
        pairs.append(_qa(
            f"Compute d/d{x} ||{x} - {b}||^2.",
            f"d/d{x} ||{x} - {b}||^2 = 2({x} - {b}). Expand: ({x}-{b})^T({x}-{b}) = {x}^T{x} - 2{b}^T{x} + {b}^T{b}, gradient is 2{x} - 2{b}."
        ))

    # OLS gradient variants
    for loss_scale in ["(1/n)", "(1/2n)", "(1/2)", ""]:
        scale_str = loss_scale + " " if loss_scale else ""
        coeff = loss_scale if loss_scale else "2"
        pairs.append(_qa(
            f"What is the gradient of L = {loss_scale}||Xw - y||^2 with respect to w?",
            f"∇_w L = {loss_scale.replace('(','').replace(')','') or '2'} X^T(Xw - y). "
            f"Setting to zero gives the normal equations X^T X w = X^T y, solved by w* = (X^T X)^{{-1}} X^T y."
        ))

    # Trace derivative facts
    trace_facts = [
        ("d/dA tr(AB)", "B^T", "d tr(AB)/dA_{ij} = B_{ji}, so the matrix derivative is B^T."),
        ("d/dA tr(A^T B)", "B", "d tr(A^T B)/dA_{ij} = B_{ij}, so the derivative is B."),
        ("d/dA tr(A)", "I", "d tr(A)/dA_{ij} = δ_{ij}, so the derivative is the identity matrix I."),
        ("d/dA tr(ABA^T) for symmetric B", "2AB", "Using the chain rule: d tr(ABA^T)/dA = A(B+B^T) = 2AB when B is symmetric."),
        ("d/dA log det(A)", "A^{-T}", "d log det(A)/dA = A^{-T}. For symmetric A this is A^{-1} (Jacobi's formula)."),
    ]
    for expr, result, explanation in trace_facts:
        pairs.append(_qa(f"What is {expr}?", f"Result: {result}. {explanation}"))

    # Eigenvalue numerical examples
    for lam1, lam2 in [(2, 3), (1, 4), (-1, 5), (2, 8), (3, 6)]:
        pairs.append(_qa(
            f"A 2×2 matrix A has eigenvalues λ₁={lam1} and λ₂={lam2}. What is tr(A) and det(A)?",
            f"tr(A) = λ₁ + λ₂ = {lam1} + {lam2} = {lam1+lam2}. "
            f"det(A) = λ₁ × λ₂ = {lam1} × {lam2} = {lam1*lam2}."
        ))
    for lam1, lam2, lam3 in [(1,2,3),(2,3,5),(1,4,6),(-1,2,4)]:
        pairs.append(_qa(
            f"A 3×3 matrix has eigenvalues {lam1},{lam2},{lam3}. What is its trace and determinant?",
            f"tr = {lam1}+{lam2}+{lam3} = {lam1+lam2+lam3}. det = {lam1}×{lam2}×{lam3} = {lam1*lam2*lam3}."
        ))

    # 2×2 determinant
    for a, b, c, d in [(1,2,3,4),(2,0,0,3),(1,1,1,1),(3,-1,2,4),(5,2,3,1),(4,3,2,1)]:
        det = a*d - b*c
        pairs.append(_qa(
            f"Compute det([[{a},{b}],[{c},{d}]]).",
            f"det = {a}×{d} - {b}×{c} = {a*d} - {b*c} = {det}."
        ))
    # 3×3 det
    for (a,b,c,d,e,f,g,h,i) in [(1,0,0,0,1,0,0,0,1),(2,1,0,1,2,1,0,1,2),(1,2,3,4,5,6,7,8,9)]:
        det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
        pairs.append(_qa(
            f"Compute the determinant of [[{a},{b},{c}],[{d},{e},{f}],[{g},{h},{i}]].",
            f"Cofactor expansion along row 1: "
            f"{a}×({e}×{i}-{f}×{h}) - {b}×({d}×{i}-{f}×{g}) + {c}×({d}×{h}-{e}×{g}) = {det}."
        ))

    # SVD and PCA conceptual
    pairs += [
        _qa("What is SVD? Write the decomposition of an m×n matrix A.",
            "A = UΣV^T where: U is m×m orthogonal (left singular vectors, columns = u_i), "
            "Σ is m×n diagonal with singular values σ_1 ≥ σ_2 ≥ ... ≥ 0, "
            "V is n×n orthogonal (right singular vectors, columns = v_i). "
            "Rank of A equals number of nonzero singular values."),
        _qa("How is PCA related to SVD of the data matrix X?",
            "Center X: X̃ = X - mean(X). Compute SVD: X̃ = UΣV^T. "
            "Principal components are columns of V. "
            "Scores (projected data) = X̃V = UΣ. "
            "Variance explained by i-th PC = σ_i^2 / Σ_j σ_j^2."),
        _qa("What is the Moore-Penrose pseudoinverse A^+ in terms of SVD?",
            "If A = UΣV^T, then A^+ = VΣ^+U^T where Σ^+ replaces each σ_i > 0 with 1/σ_i. "
            "For full-rank A: A^+ = (A^T A)^{-1} A^T (left pseudoinverse). "
            "The least-squares solution is w* = A^+ b."),
        _qa("What is the rank-k approximation of matrix A and why is it optimal?",
            "A_k = Σ_{i=1}^{k} σ_i u_i v_i^T. "
            "By the Eckart-Young theorem: A_k minimises ||A - B||_F over all rank-k matrices B. "
            "Used in dimensionality reduction, image compression, and collaborative filtering."),
        _qa("What is the relationship between Frobenius norm and singular values?",
            "||A||_F = sqrt(Σ_i σ_i^2) = sqrt(tr(A^T A)). "
            "The Frobenius norm equals the Euclidean norm of the vector of singular values."),
        _qa("What is an orthogonal matrix Q and what are its properties?",
            "Q is orthogonal if Q^T Q = Q Q^T = I. Properties: "
            "det(Q) = ±1; singular values are all 1; preserves vector norms (||Qx|| = ||x||); "
            "preserves dot products (Qx)·(Qy) = x·y. "
            "Rotation and reflection matrices are orthogonal."),
    ]
    return pairs


def gen_probability(rng):
    pairs = []

    # Bayes theorem numerical
    for p_pos, p_dis, p_pos_given_dis in [(0.99, 0.001, 0.99), (0.95, 0.01, 0.95), (0.9, 0.1, 0.8)]:
        p_pos_given_no_dis = p_pos - p_pos_given_dis * p_dis  # rough
        pairs.append(_qa(
            f"A disease has prevalence {p_dis}. A test has sensitivity (true positive rate) {p_pos_given_dis} "
            f"and false positive rate {round(1-p_pos,2)}. What is the positive predictive value?",
            f"By Bayes: P(disease|+) = P(+|disease)P(disease) / P(+). "
            f"P(+) = P(+|dis)×P(dis) + P(+|no dis)×P(no dis) = {p_pos_given_dis}×{p_dis} + {round(1-p_pos,2)}×{round(1-p_dis,3)}. "
            f"PPV = {p_pos_given_dis}×{p_dis} / P(+)."
        ))

    # MLE - Gaussian mean
    for n, mean, var in [(10, 3.0, 1.0), (100, 5.0, 2.0), (50, 0.0, 1.0)]:
        pairs.append(_qa(
            f"You observe n={n} i.i.d. samples from N(μ, {var}). What is the MLE for μ?",
            f"Log-likelihood: ℓ(μ) = -n/(2×{var}) Σ(x_i - μ)² + const. "
            f"∂ℓ/∂μ = (1/{var}) Σ(x_i - μ) = 0 → μ_MLE = (1/n) Σ x_i = x̄ (sample mean)."
        ))

    # MLE - Bernoulli
    for n, k in [(10,3),(100,42),(50,25),(200,80)]:
        p_hat = k/n
        pairs.append(_qa(
            f"You observe {k} successes in {n} Bernoulli trials. What is the MLE for p?",
            f"Log-likelihood: ℓ(p) = {k} log p + {n-k} log(1-p). "
            f"dℓ/dp = {k}/p - {n-k}/(1-p) = 0 → p_MLE = {k}/{n} = {fmt_float(p_hat)}."
        ))

    # MLE - Poisson
    for n, sum_x in [(10,35),(50,200),(100,300)]:
        lam = sum_x/n
        pairs.append(_qa(
            f"Given {n} i.i.d. Poisson(λ) observations summing to {sum_x}, what is λ_MLE?",
            f"Log-likelihood: ℓ(λ) = Σx_i log λ - nλ - const. "
            f"dℓ/dλ = Σx_i/λ - n = 0 → λ_MLE = Σx_i/n = {sum_x}/{n} = {fmt_float(lam)}."
        ))

    # Entropy calculations
    for probs, name in [([0.5,0.5],"fair coin"), ([0.25,0.25,0.25,0.25],"4-sided die"), ([1.0],"deterministic")]:
        h = -sum(p*math.log2(p) for p in probs if p > 0)
        pairs.append(_qa(
            f"Compute the entropy H(X) in bits for a {name} with probabilities {probs}.",
            f"H(X) = -Σ p log₂(p) = -{' - '.join(f'{p}×log₂({p})' for p in probs if p>0)} = {fmt_float(h)} bits."
        ))

    # KL divergence
    pairs += [
        _qa("What is KL divergence KL(P||Q) and is it symmetric?",
            "KL(P||Q) = Σ P(x) log(P(x)/Q(x)). KL is NOT symmetric: KL(P||Q) ≠ KL(Q||P). "
            "KL ≥ 0 always (Gibbs inequality). KL = 0 iff P = Q."),
        _qa("Prove KL(P||Q) ≥ 0.",
            "By Jensen's inequality (applied to convex -log): "
            "KL(P||Q) = -E_P[log(Q/P)] ≥ -log(E_P[Q/P]) = -log(Σ_x P(x)·Q(x)/P(x)) = -log(1) = 0."),
        _qa("Compute KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) for two Gaussians.",
            "KL = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2. "
            "Special case KL(N(μ,1)||N(0,1)) = μ²/2. "
            "Used in VAE regularisation term."),
        _qa("What is the relationship H(P,Q) = H(P) + KL(P||Q)?",
            "Cross-entropy H(P,Q) = -Σ P(x) log Q(x) = -Σ P(x) log P(x) + Σ P(x) log(P(x)/Q(x)) "
            "= H(P) + KL(P||Q). Minimising cross-entropy w.r.t. Q (with P fixed) minimises KL(P||Q)."),
    ]

    # MAP vs MLE
    for prior in ["N(0,1)", "N(0,λ⁻¹)"]:
        pairs.append(_qa(
            f"With Gaussian prior {prior} on weights, what does MAP estimation give?",
            f"MAP = argmax [log P(data|w) + log P(w)] = argmax [log likelihood - λ/2 ||w||²]. "
            f"This is equivalent to L2-regularised MLE (Ridge regression). "
            f"The L2 penalty term comes directly from the Gaussian log-prior."
        ))

    # Bias-variance
    pairs += [
        _qa("State the bias-variance decomposition of MSE for estimator f̂.",
            "E[(y - f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ² (noise). "
            "Bias = E[f̂(x)] - f(x) (systematic error). "
            "Variance = E[(f̂(x) - E[f̂(x)])²] (sensitivity to training set). "
            "Increasing model complexity typically reduces bias but increases variance."),
        _qa("What is the variance of the sample mean X̄ = (1/n)Σ X_i for i.i.d. X_i with Var(X_i)=σ²?",
            "Var(X̄) = Var((1/n)Σ X_i) = (1/n²) Σ Var(X_i) = (1/n²)(nσ²) = σ²/n. "
            "Standard error = σ/√n — reduces with more data."),
        _qa("State the central limit theorem.",
            "For i.i.d. X_i with mean μ and variance σ², "
            "√n(X̄ - μ) → N(0, σ²) in distribution as n → ∞. "
            "Justifies Gaussian approximations and forms the basis of confidence intervals."),
    ]
    return pairs


def gen_loss_functions(rng):
    pairs = []

    # MSE gradient with various formulations
    for scale, result in [("(1/n)", "(2/n)X^T(Xw-y)"), ("(1/2n)", "(1/n)X^T(Xw-y)"),
                          ("(1/2)", "X^T(Xw-y)"), ("", "2X^T(Xw-y)")]:
        pairs.append(_qa(
            f"Compute ∇_w L where L = {scale}||Xw - y||².",
            f"∇_w L = {result}. Setting to zero: X^T X w = X^T y (normal equations)."
        ))

    # Binary cross-entropy gradient through logit
    pairs += [
        _qa("Derive dL/dz for binary cross-entropy L = -[y log σ(z) + (1-y) log(1-σ(z))].",
            "Let s = σ(z) = 1/(1+e^{-z}), ds/dz = s(1-s). "
            "dL/dz = -y/s · s(1-s) + (1-y)/(1-s) · s(1-s) = -y(1-s) + (1-y)s = s - y = σ(z) - y. "
            "Gradient = prediction − label."),
        _qa("What is the gradient of sigmoid σ(z) = 1/(1+e^{-z})?",
            "dσ/dz = σ(z)(1 - σ(z)). Maximum at z=0 where σ=0.5, giving gradient 0.25. "
            "For large |z|, gradient → 0 (saturation)."),
        _qa("What is the gradient of tanh(z)?",
            "d tanh(z)/dz = 1 - tanh²(z) = sech²(z). "
            "Range: (0,1]. Max gradient 1 at z=0. Saturates for large |z| like sigmoid."),
    ]

    # Softmax + CE gradient
    pairs += [
        _qa("Derive the gradient of categorical cross-entropy L = -Σ_i y_i log s_i w.r.t. logits z.",
            "s = softmax(z). ∂L/∂z_i = Σ_j (-y_j/s_j) ∂s_j/∂z_i. "
            "Using ∂s_j/∂z_i = s_j(δ_{ij} - s_i): ∂L/∂z_i = s_i - y_i. "
            "In vector form: ∂L/∂z = s - y (predictions minus one-hot labels)."),
        _qa("What is the softmax Jacobian ∂s_i/∂z_j?",
            "∂s_i/∂z_j = s_i(δ_{ij} - s_j). "
            "Matrix form: J = diag(s) - s s^T. "
            "J is symmetric, positive semi-definite, and has a zero eigenvalue (s is in its null space)."),
        _qa("Why does softmax+CE have such a clean gradient?",
            "The gradient ∂L/∂z = s - y happens because softmax's Jacobian exactly cancels "
            "the 1/s terms from the log: Σ_j y_j ∂s_j/∂z_i = Σ_j y_j s_j(δ_{ij}-s_i) = s_i(y_i-1) + s_i Σ_j y_j = s_i - y_i."),
    ]

    # Numerical BCE examples
    for y, z, s in [(1, 2.0, 0.880), (0, -1.0, 0.269), (1, 0.0, 0.5)]:
        grad = round(s - y, 4)
        pairs.append(_qa(
            f"For binary cross-entropy with y={y}, logit z={z}, σ(z)≈{s}, what is dL/dz?",
            f"dL/dz = σ(z) - y = {s} - {y} = {grad}."
        ))

    # Loss landscape
    pairs += [
        _qa("What is the hinge loss and its subgradient for binary classification?",
            "L_hinge = max(0, 1 - y·f(x)) for y ∈ {-1,+1}. "
            "Subgradient: 0 if y·f(x) ≥ 1 (correct with margin), else -y·x (push in correct direction). "
            "Used in SVMs. Provides margin maximisation unlike logistic loss."),
        _qa("Compare L1 and L2 regularisation effects on learned weights.",
            "L2 (Ridge): adds λ||w||² to loss. Gradient: 2λw → weights shrink proportionally (never exactly zero). "
            "L1 (Lasso): adds λ||w||₁. Subgradient: λ·sign(w) → induces exact sparsity. "
            "Elastic net combines both: λ₁||w||₁ + λ₂||w||². "
            "L2 corresponds to Gaussian prior; L1 to Laplace prior (MAP interpretation)."),
        _qa("What is the logistic regression loss and its connection to MLE?",
            "L = -Σ[y_i log σ(w^T x_i) + (1-y_i) log(1-σ(w^T x_i))]. "
            "This is the negative log-likelihood under Bernoulli model with p = σ(w^T x). "
            "Logistic regression = MLE for a Bernoulli GLM with logit link function."),
        _qa("What is the gradient of log-sum-exp? Why use it instead of log(Σ exp(z_i))?",
            "d/dz_i log(Σ_j exp(z_j)) = exp(z_i) / Σ_j exp(z_j) = softmax(z)_i. "
            "Numerically stable computation: LSE(z) = z_max + log Σ exp(z_j - z_max). "
            "Prevents overflow in exp without changing the value (max cancels in gradient)."),
    ]
    return pairs


def gen_backprop(rng):
    pairs = []

    # Linear layer backprop with different notation
    for x_name, w_name in [("x","W"),("h","U"),("a","A")]:
        pairs.append(_qa(
            f"For the linear layer y = {w_name}{x_name} + b, derive ∂L/∂{w_name}, ∂L/∂b, ∂L/∂{x_name}.",
            f"Let δ = ∂L/∂y (upstream gradient). "
            f"∂L/∂{w_name} = δ {x_name}^T (outer product). "
            f"∂L/∂b = δ. "
            f"∂L/∂{x_name} = {w_name}^T δ (backpropagate through weight matrix)."
        ))

    # Chain rule / backprop conceptual
    pairs += [
        _qa("Explain backpropagation using the chain rule for f = f_L ∘ ... ∘ f_1.",
            "Forward pass: x_k = f_k(x_{k-1}). "
            "Backward pass: δ_k = (∂f_{k+1}/∂x_k)^T δ_{k+1} where δ_L = ∂L/∂x_L. "
            "Parameter gradient: ∂L/∂θ_k = (∂f_k/∂θ_k)^T δ_k. "
            "Backprop is just the chain rule applied layer by layer in reverse."),
        _qa("What is the time complexity of backpropagation relative to forward pass?",
            "Backprop has the same O(n) complexity as the forward pass (up to constant factor). "
            "This is because backprop reuses activations from the forward pass and applies "
            "one Jacobian-vector product per layer — each is as cheap as the forward linear op."),
        _qa("Why is gradient checkpointing useful and what is the tradeoff?",
            "Gradient checkpointing recomputes activations during backprop instead of storing them. "
            "Memory: O(√n) instead of O(n) for n layers. "
            "Compute: ~33% more FLOPs (forward pass run twice). "
            "Enables training much larger models on the same GPU memory."),
    ]

    # Activation gradients (many variants)
    for activation, formula, grad_formula in [
        ("ReLU", "max(0,x)", "1 if x>0 else 0"),
        ("Leaky ReLU", "max(αx,x)", "1 if x>0 else α"),
        ("ELU", "x if x>0 else α(e^x-1)", "1 if x>0 else αe^x"),
        ("GELU", "x·Φ(x) (approx: x·σ(1.702x))", "Φ(x) + x·φ(x) ≈ σ(1.702x) + 1.702x·σ(1.702x)(1-σ(1.702x))"),
        ("Swish", "x·σ(x)", "σ(x) + x·σ(x)(1-σ(x)) = σ(x)(1 + x(1-σ(x)))"),
        ("Sigmoid", "1/(1+e^{-x})", "σ(x)(1-σ(x))"),
    ]:
        pairs.append(_qa(
            f"What is the gradient of the {activation} activation function?",
            f"{activation}(x) = {formula}. Gradient: d({activation})/dx = {grad_formula}."
        ))

    # Attention
    pairs += [
        _qa("Write the scaled dot-product attention formula and state its complexity.",
            "Attention(Q,K,V) = softmax(QK^T / √d_k) V. "
            "Shapes: Q,K ∈ R^{n×d_k}, V ∈ R^{n×d_v}, output ∈ R^{n×d_v}. "
            "Complexity: O(n² d_k) time, O(n²) memory (attention matrix). "
            "Bottleneck is the n×n attention matrix — quadratic in sequence length."),
        _qa("Why scale attention by 1/√d_k?",
            "q^T k = Σ_{i=1}^{d_k} q_i k_i. If q_i,k_i ~ N(0,1): Var(q^T k) = d_k. "
            "Large d_k → large dot products → softmax saturates → vanishing gradients. "
            "Dividing by √d_k normalises variance to 1 regardless of d_k."),
        _qa("What is multi-head attention?",
            "MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O where "
            "head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V). "
            "Each head projects to d_k = d_model/h dimensions. "
            "Multiple heads let the model attend to different positions/features jointly."),
        _qa("Derive the gradient of attention output w.r.t. the query Q.",
            "Let A = softmax(QK^T/√d_k), output = AV. "
            "∂L/∂Q = (1/√d_k) · (∂L/∂A ∘ A - A·diag(∂L/∂A ∘ A · 1)) · K "
            "where the middle term is the softmax Jacobian. In practice computed by autograd."),
        _qa("What is layer normalisation and how does it differ from batch normalisation?",
            "LayerNorm: normalise over the feature dimension for each token independently: "
            "y = γ(x-μ)/σ + β where μ,σ computed over d features. "
            "BatchNorm: normalise over the batch dimension per feature: "
            "requires large batch; statistics differ between train (batch) and test (running avg). "
            "LN is preferred in transformers because it works with any batch size including 1."),
    ]

    # Initialisation
    pairs += [
        _qa("What is Xavier/Glorot initialisation and why is it used?",
            "Var(w) = 2/(n_in + n_out). Derived by requiring Var(output) ≈ Var(input) "
            "both in forward and backward pass for tanh/sigmoid activations. "
            "Prevents activations from exploding or vanishing through depth."),
        _qa("What is He initialisation and when should it be used instead of Xavier?",
            "Var(w) = 2/n_in. Designed for ReLU activations (which zero out half their inputs, "
            "effectively halving the variance). Using Xavier with ReLU leads to vanishing activations. "
            "He init compensates by doubling the variance."),
        _qa("What is the dying ReLU problem?",
            "If a ReLU unit's pre-activation is always negative, it outputs 0 and has gradient 0. "
            "The unit never recovers — it is permanently dead. "
            "Causes: large negative bias, high learning rate. "
            "Fixes: Leaky ReLU (gradient α<1 for x<0), ELU, careful LR tuning."),
    ]
    return pairs


def gen_optimisation(rng):
    pairs = []

    # GD numerical examples
    for w0, lr, grad in [(2.0, 0.1, 5.0), (1.0, 0.01, 3.0), (0.5, 0.001, 10.0), (3.0, 0.05, 4.0)]:
        w1 = round(w0 - lr*grad, 6)
        pairs.append(_qa(
            f"Gradient descent: w={w0}, α={lr}, ∂L/∂w={grad}. What is the next w?",
            f"w_new = w - α·g = {w0} - {lr}×{grad} = {w1}."
        ))

    # Adam numerical example
    for g, beta1, beta2 in [(0.5, 0.9, 0.999), (1.0, 0.9, 0.999), (0.3, 0.9, 0.999)]:
        m1 = round((1-beta1)*g, 4)
        v1 = round((1-beta2)*g**2, 6)
        m_hat = round(m1/(1-beta1), 4)
        v_hat = round(v1/(1-beta2), 4)
        pairs.append(_qa(
            f"Adam update at t=1: g={g}, β₁={beta1}, β₂={beta2}, ε=1e-8. Compute m̂, v̂.",
            f"m₁ = (1-β₁)g = {m1}. v₁ = (1-β₂)g² = {v1}. "
            f"m̂₁ = m₁/(1-β₁^1) = {m1}/{round(1-beta1,1)} = {m_hat}. "
            f"v̂₁ = v₁/(1-β₂^1) = {v1}/{round(1-beta2,3)} = {v_hat}."
        ))

    # Convergence theory
    pairs += [
        _qa("What is the gradient descent convergence rate for L-smooth convex f with step α=1/L?",
            "f(w_t) - f* ≤ ||w_0 - w*||² L / (2t). Convergence rate O(1/t). "
            "Need O(L/ε) steps to reach ε-accuracy."),
        _qa("What is the gradient descent convergence rate for μ-strongly convex, L-smooth f?",
            "f(w_t) - f* ≤ (1 - μ/L)^t (f(w_0) - f*). Linear (geometric) convergence. "
            "Condition number κ = L/μ; rate = (1-1/κ)^t. Need O(κ log(1/ε)) steps."),
        _qa("What is L-smoothness? State the descent lemma.",
            "f is L-smooth if ||∇f(x)-∇f(y)|| ≤ L||x-y||. Equivalent to: Hessian eigenvalues ≤ L. "
            "Descent lemma: f(y) ≤ f(x) + ∇f(x)^T(y-x) + (L/2)||y-x||². "
            "With step α=1/L: f(w_{t+1}) ≤ f(w_t) - (1/(2L))||∇f(w_t)||². "
            "Guarantees monotone decrease in loss."),
        _qa("What is gradient clipping? Why is it important?",
            "If ||g|| > threshold c: g ← g × c/||g||. "
            "Prevents exploding gradients in RNNs and transformers. "
            "Triggered by occasional large gradients from long-range backprop or bad batches. "
            "Does NOT prevent vanishing gradients (different problem)."),
        _qa("What is the difference between gradient descent, SGD, and mini-batch SGD?",
            "Full-batch GD: exact gradient, O(n) per step, deterministic. "
            "SGD (B=1): noisy gradient, O(1) per step, implicit regularisation, can escape local minima. "
            "Mini-batch SGD (B>1): unbiased gradient estimate, GPU-parallelisable, lower variance than SGD. "
            "In practice, mini-batch is used. 'SGD' in frameworks usually means mini-batch SGD."),
        _qa("What is momentum and how does it accelerate convergence?",
            "v_t = γ v_{t-1} + α ∇L(w_t); w_{t+1} = w_t - v_t. "
            "Effect: accumulates velocity in consistent gradient directions, damps oscillations. "
            "For strongly convex f, momentum achieves O((√κ - 1)/(√κ + 1))^t rate vs O((κ-1)/(κ+1))^t "
            "for plain GD — √κ improvement in condition number dependence."),
        _qa("Describe the Adam optimiser. What are m_t and v_t?",
            "m_t = β₁ m_{t-1} + (1-β₁)g_t (exponential moving average of gradients, 1st moment). "
            "v_t = β₂ v_{t-1} + (1-β₂)g_t² (exponential moving average of squared gradients, 2nd moment). "
            "Bias correction: m̂_t = m_t/(1-β₁^t), v̂_t = v_t/(1-β₂^t). "
            "Update: w_{t+1} = w_t - α m̂_t/(√v̂_t + ε). "
            "Adapts LR per-parameter; good for sparse gradients and noisy problems."),
        _qa("What is learning rate warmup and why do transformers use it?",
            "Warmup: linearly increase lr from 0 to lr_max over warmup_steps, then decay. "
            "Early training: model weights are random, gradients are unreliable, large lr → instability. "
            "Warmup gives the model time to reach a stable region before large updates. "
            "Cosine schedule: lr(t) = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))."),
        _qa("What is the condition number of a matrix and how does it affect optimisation?",
            "κ(A) = σ_max(A)/σ_min(A) = ||A||_2 ||A^{-1}||_2. "
            "For quadratic f(w) = (1/2)||Aw-b||², GD convergence rate = ((κ-1)/(κ+1))^t. "
            "High κ: loss surface is elongated ellipse → gradient zig-zags → slow convergence. "
            "Solution: preconditioning (multiply gradient by approx Hessian inverse)."),
    ]
    return pairs


def gen_information_theory(rng):
    pairs = []

    # Entropy with different distributions
    for n in [2, 4, 8, 16, 32]:
        h = math.log2(n)
        pairs.append(_qa(
            f"What is the entropy in bits of a uniform distribution over {n} outcomes?",
            f"H = log₂({n}) = {h} bits. Uniform distribution maximises entropy for a given support."
        ))

    pairs += [
        _qa("What is mutual information I(X;Y)?",
            "I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y) = KL(P(X,Y) || P(X)P(Y)). "
            "Measures reduction in uncertainty about X given Y. I(X;Y) = 0 iff X ⊥ Y."),
        _qa("State the data processing inequality.",
            "If X → Y → Z is a Markov chain, I(X;Z) ≤ I(X;Y). "
            "Processing cannot increase mutual information. "
            "Neural network representations cannot capture more info about X than the input does."),
        _qa("What is the connection between MLE, cross-entropy, and KL divergence?",
            "MLE: max_θ Σ log p_θ(x_i) = min_θ H(p_data, p_θ). "
            "Since H(p,q) = H(p) + KL(p||q) and H(p_data) is fixed, "
            "MLE also minimises KL(p_data || p_θ). All three objectives are equivalent."),
        _qa("What is perplexity in language modelling?",
            "PPL = exp(H(p_data, p_model)) = exp(-1/N Σ_i log p(x_i)). "
            "Perplexity k means the model is as uncertain as a uniform distribution over k choices. "
            "Lower is better. Relation to loss: PPL = exp(cross-entropy loss)."),
        _qa("What is the maximum entropy principle?",
            "Among all distributions satisfying given constraints, choose the one with maximum entropy. "
            "This is the least informative (most unbiased) choice. "
            "Example: with only mean and variance constraints, the max-entropy distribution is Gaussian."),
        _qa("What is the ELBO (evidence lower bound) in variational inference?",
            "log p(x) = ELBO + KL(q(z)||p(z|x)) ≥ ELBO. "
            "ELBO = E_q[log p(x|z)] - KL(q(z)||p(z)). "
            "Maximising ELBO tightens the bound and minimises KL(q||p(z|x)). "
            "VAE: decoder = log p(x|z), encoder = q(z|x), prior = p(z) = N(0,I)."),
        _qa("Derive the reparameterisation trick used in VAEs.",
            "Sampling z ~ N(μ,σ²) is not differentiable w.r.t. μ,σ. "
            "Reparameterisation: z = μ + σ·ε, ε ~ N(0,1). "
            "Now z is a deterministic function of (μ,σ,ε), so gradients flow through μ and σ. "
            "This enables backprop through the sampling operation."),
    ]
    return pairs


def gen_numerical_la(rng):
    """Numerical linear algebra problems with concrete step-by-step solutions."""
    pairs = []

    # Gradient descent steps — large grid
    for w0 in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        for lr in [0.001, 0.01, 0.1, 0.5]:
            for grad in [1.0, 2.0, 5.0, 10.0]:
                w1 = round(w0 - lr*grad, 6)
                pairs.append(_qa(
                    f"Gradient descent: w={w0}, α={lr}, ∂L/∂w={grad}. Compute the next weight.",
                    f"w_new = w - α·g = {w0} - {lr}×{grad} = {w1}."
                ))

    # GD multiple steps
    for w0, lr, grad in [(2.0, 0.1, 3.0), (5.0, 0.01, 4.0), (1.0, 0.5, 0.5)]:
        steps = []
        w = w0
        for t in range(1, 4):
            w = round(w - lr*grad, 6)
            steps.append(f"step {t}: w={w}")
        pairs.append(_qa(
            f"GD with w₀={w0}, α={lr}, constant gradient g={grad}. Show 3 steps.",
            "; ".join(steps) + f". Pattern: w_t = {w0} - t×{lr}×{grad}."
        ))

    # L-smooth optimal LR
    for L in [1.0, 2.0, 5.0, 10.0, 100.0]:
        pairs.append(_qa(
            f"For an L={L}-smooth function, what is the optimal GD step size?",
            f"α* = 1/L = {fmt_float(1/L)}. This guarantees monotone decrease: "
            f"f(w_{{t+1}}) ≤ f(w_t) - 1/(2×{L})||∇f(w_t)||²."
        ))

    # Condition number → convergence
    for kappa in [2, 5, 10, 50, 100]:
        rate = round((kappa-1)/(kappa+1), 4)
        pairs.append(_qa(
            f"A quadratic has condition number κ={kappa}. What is the GD convergence rate?",
            f"GD convergence rate: ((κ-1)/(κ+1))^t = (({kappa}-1)/({kappa}+1))^t = {rate}^t. "
            f"Need O(κ log(1/ε)) = O({kappa} log(1/ε)) steps for ε-accuracy."
        ))

    # Power iteration convergence
    for lam1, lam2 in [(3,1),(5,2),(10,3),(4,1),(8,2),(20,5)]:
        ratio = round(lam2/lam1, 4)
        pairs.append(_qa(
            f"Power iteration on a matrix with top eigenvalues λ₁={lam1} and λ₂={lam2}. Rate?",
            f"Convergence rate (|λ₂/λ₁|)^t = ({lam2}/{lam1})^t = {ratio}^t. "
            f"After t steps, error is O({ratio}^t). {'Converges quickly.' if ratio < 0.5 else 'Slow — eigenvalues are close.'}"
        ))

    # Softmax — many examples
    for zs in [(1,2,3),(0,0,0),(1,1,2),(2,4,6),(0,1,0),(3,1,2),(0,0,1),(5,5,5)]:
        exps = [math.exp(z) for z in zs]
        total = sum(exps)
        softmax = [round(e/total, 4) for e in exps]
        pairs.append(_qa(
            f"Compute softmax({list(zs)}).",
            f"exp values: {[round(e,3) for e in exps]}. Sum = {round(total,3)}. "
            f"softmax = {softmax}. Verify: sum = {round(sum(softmax),4)}."
        ))

    # 3-class softmax + CE gradient
    for z1,z2,z3,y_class in [(2,1,0,0),(1,2,1,1),(0,0,3,2),(3,2,1,0)]:
        zs = [z1,z2,z3]
        exps = [math.exp(z) for z in zs]
        total = sum(exps)
        s = [round(e/total,4) for e in exps]
        y = [1 if i==y_class else 0 for i in range(3)]
        grad = [round(s[i]-y[i],4) for i in range(3)]
        pairs.append(_qa(
            f"Logits z={zs}, true class={y_class}. Compute softmax and CE gradient ∂L/∂z.",
            f"softmax = {s}. One-hot y={y}. "
            f"∂L/∂z = s - y = {grad}."
        ))

    # Sigmoid — many z values
    for z in [-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4]:
        s = round(1/(1+math.exp(-z)), 4)
        ds = round(s*(1-s), 4)
        pairs.append(_qa(
            f"Compute σ({z}) and its gradient dσ/dz.",
            f"σ({z}) = 1/(1+e^{{-{z}}}) = {s}. Gradient: σ(1-σ) = {s}×{round(1-s,4)} = {ds}."
        ))

    # ReLU backprop numerical
    for x, upstream in [(-1.0, 0.5),(0.5, 2.0),(-0.1, 1.0),(3.0, 0.3),(0.0, 1.0)]:
        local_grad = 1.0 if x > 0 else 0.0
        back = round(local_grad * upstream, 4)
        pairs.append(_qa(
            f"ReLU backprop: pre-activation x={x}, upstream gradient δ={upstream}. Compute ∂L/∂x.",
            f"ReLU gradient: 1{{x>0}} = {int(local_grad)}. ∂L/∂x = δ × 1{{x>0}} = {upstream} × {int(local_grad)} = {back}."
        ))

    # MSE gradient numerical
    for w, X_norm2, Xw_minus_y_norm in [(1.0, 4.0, 2.0),(0.5, 9.0, 1.5),(2.0, 1.0, 0.5)]:
        grad = round(2 * X_norm2 * w - 2 * Xw_minus_y_norm, 4)
        pairs.append(_qa(
            f"1D case: L = (Xw-y)², X²={X_norm2}, Xw-y={Xw_minus_y_norm}. Compute ∂L/∂w.",
            f"∂L/∂w = 2X(Xw-y) = 2×{math.sqrt(X_norm2):.2f}×{Xw_minus_y_norm} = {round(2*math.sqrt(X_norm2)*Xw_minus_y_norm,4)}."
        ))

    # Adam step
    for g, lr in [(0.5, 0.001),(1.0, 0.001),(0.3, 0.01),(2.0, 0.0001)]:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = round((1-beta1)*g, 6)
        v = round((1-beta2)*g**2, 8)
        m_hat = round(m/(1-beta1), 4)
        v_hat = round(v/(1-beta2), 6)
        update = round(lr * m_hat / (v_hat**0.5 + eps), 6)
        pairs.append(_qa(
            f"Adam t=1: g={g}, α={lr}, β₁=0.9, β₂=0.999. Compute m̂₁, v̂₁, and weight update.",
            f"m₁=(1-0.9)×{g}={m}. v₁=(1-0.999)×{g}²={v}. "
            f"m̂₁={m}/{1-beta1}={m_hat}. v̂₁={v}/{1-beta2}={v_hat}. "
            f"Δw = α×m̂/√v̂ = {lr}×{m_hat}/√{v_hat} ≈ {update}."
        ))

    # Entropy numerical
    for n in [2,4,8,16,32,64]:
        h = math.log2(n)
        pairs.append(_qa(
            f"What is the entropy of a uniform distribution over {n} events?",
            f"H = -Σ (1/{n}) log₂(1/{n}) = log₂({n}) = {h:.1f} bits. "
            f"Uniform distribution maximises entropy for {n} outcomes."
        ))

    # KL Gaussian numerical
    for mu1, mu2, s1, s2 in [(0,1,1,1),(0,0,1,2),(1,0,1,1),(2,0,1,1)]:
        kl = round(math.log(s2/s1) + (s1**2 + (mu1-mu2)**2)/(2*s2**2) - 0.5, 4)
        pairs.append(_qa(
            f"Compute KL(N({mu1},{s1}²) || N({mu2},{s2}²)).",
            f"KL = log({s2}/{s1}) + ({s1}² + ({mu1}-{mu2})²)/(2×{s2}²) - 1/2 "
            f"= {round(math.log(s2/s1),4)} + {round((s1**2+(mu1-mu2)**2)/(2*s2**2),4)} - 0.5 = {kl}."
        ))

    # Perplexity
    for loss in [1.0, 2.0, 3.0, 0.5, 4.0]:
        ppl = round(math.exp(loss), 2)
        pairs.append(_qa(
            f"A language model has cross-entropy loss {loss} nats. What is its perplexity?",
            f"PPL = exp(loss) = exp({loss}) ≈ {ppl}. "
            f"This means the model is as uncertain as a uniform distribution over ~{int(ppl)} choices."
        ))

    return pairs



def gen_attention_math(rng):
    """Scaled dot-product attention, multi-head, masking, KV cache math."""
    pairs = []

    # Scaled dot-product attention formula variants
    attn_templates = [
        ("What is the formula for scaled dot-product attention?",
         "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. "
         "Dividing by sqrt(d_k) prevents dot products from growing large in high dimensions, "
         "which would push softmax into near-zero gradient regions."),
        ("Why do we divide by sqrt(d_k) in attention?",
         "For random Q and K with unit variance, QK^T has variance d_k. "
         "Dividing by sqrt(d_k) keeps variance ≈ 1, preventing softmax saturation. "
         "At high d_k, large logits push softmax outputs near 0 or 1, killing gradients."),
        ("What is the gradient of softmax(z) with respect to z?",
         "∂softmax(z)_i / ∂z_j = softmax(z)_i (δ_{ij} - softmax(z)_j). "
         "In matrix form: ∂s/∂z = diag(s) - s s^T, where s = softmax(z)."),
        ("In multi-head attention with h heads, d_model dimensions, what are Q, K, V sizes per head?",
         "d_k = d_v = d_model / h. Each head projects Q to (seq, d_k), K to (seq, d_k), V to (seq, d_v). "
         "Total parameters for W_Q, W_K, W_V: each is d_model × d_model split across h heads."),
        ("What is the output of multi-head attention?",
         "MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O, where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i). "
         "W_O ∈ R^{h*d_v × d_model} projects back to d_model dimensions."),
        ("What is causal (masked) self-attention and how is it implemented?",
         "Causal attention prevents position i from attending to positions j > i (future tokens). "
         "Implemented by adding -inf to the attention logits QK^T / sqrt(d_k) at positions (i,j) where j > i "
         "before applying softmax. softmax(-inf) = 0, so those positions get zero weight."),
        ("What is cross-attention in a Transformer decoder?",
         "Cross-attention computes Q from the decoder hidden state, K and V from the encoder output. "
         "This lets each decoder position attend over the full encoder sequence. "
         "Formula: Attention(Q_dec, K_enc, V_enc) = softmax(Q_dec K_enc^T / sqrt(d_k)) V_enc."),
        ("What is the computational complexity of self-attention?",
         "O(n^2 d) where n is sequence length and d is embedding dimension. "
         "The QK^T matrix multiplication is O(n^2 d_k) and attention-weighted V sum is O(n^2 d_v). "
         "This quadratic cost in n is the bottleneck for long sequences."),
    ]
    pairs.extend([_qa(q, a) for q, a in attn_templates])

    # Numerical attention examples
    for d_k in [32, 64, 128, 256, 512]:
        scale = round(1.0 / math.sqrt(d_k), 4)
        pairs.append(_qa(
            f"In attention with d_k={d_k}, what is the scaling factor 1/sqrt(d_k)?",
            f"1/sqrt({d_k}) = {scale:.4f}. The logits QK^T are multiplied by this before softmax."
        ))

    # KV cache
    for n_layers, n_heads, d_head, seq_len in [(6, 8, 64, 512), (12, 12, 64, 1024), (8, 4, 128, 512)]:
        kv_size_mb = 2 * n_layers * 2 * seq_len * n_heads * d_head * 2 / 1e6  # bfloat16
        pairs.append(_qa(
            f"A transformer with {n_layers} layers, {n_heads} heads, d_head={d_head}, seq_len={seq_len}. "
            f"How large is the KV cache in bfloat16 (2 bytes per value)?",
            f"KV cache = 2 (K and V) × {n_layers} layers × {seq_len} positions × {n_heads} heads × {d_head} dims × 2 bytes "
            f"= {kv_size_mb:.1f} MB. This grows linearly with sequence length."
        ))

    return pairs



def gen_normalization_math(rng):
    """Layer norm, batch norm, RMS norm — forward formulas and gradient intuition."""
    pairs = []

    norm_templates = [
        ("What is Layer Normalization and how is it computed?",
         "LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β, where μ = mean(x), σ² = var(x) computed per token. "
         "γ and β are learned scale and shift parameters. ε (typically 1e-5) prevents division by zero. "
         "Unlike BatchNorm, LayerNorm normalizes across the feature dimension, not the batch dimension."),
        ("What is the difference between Layer Norm and Batch Norm?",
         "BatchNorm normalizes across the batch dimension: μ_j = (1/N) Σ_i x_{ij} for each feature j. "
         "LayerNorm normalizes across the feature dimension: μ_i = (1/D) Σ_j x_{ij} for each sample i. "
         "LayerNorm is preferred in NLP because it doesn't depend on batch size and works with variable-length sequences."),
        ("What is RMSNorm and how does it differ from LayerNorm?",
         "RMSNorm(x) = x / RMS(x) * γ, where RMS(x) = sqrt((1/D) Σ x_i²). "
         "It drops the mean subtraction (centering) from LayerNorm, keeping only the scale normalization. "
         "This is cheaper (no mean computation) and works just as well in practice — used in LLaMA, GPT-NeoX."),
        ("What is the gradient of LayerNorm with respect to its input x?",
         "∂LayerNorm/∂x is complex: it involves the Jacobian of (x - μ)/σ w.r.t. x. "
         "The key insight: normalizing couples all dimensions, so ∂(x_i - μ)/∂x_j involves both the direct "
         "path (δ_{ij}) and paths through μ (-1/D) and σ² (-x̂_i * x̂_j / D). "
         "In practice, autograd handles this; the result is a D×D dense Jacobian."),
        ("Why is pre-norm (norm before attention/MLP) preferred over post-norm in deep transformers?",
         "Pre-norm: x' = x + f(LayerNorm(x)). Post-norm: x' = LayerNorm(x + f(x)). "
         "Pre-norm keeps the residual path clean — gradients flow directly through x without going through "
         "the norm operation, preventing vanishing gradients in very deep networks. "
         "Post-norm tends to require careful LR warmup; pre-norm is more stable from the start."),
    ]
    pairs.extend([_qa(q, a) for q, a in norm_templates])

    # Numerical layer norm examples
    for vals, eps in [([1.0, 2.0, 3.0, 4.0], 1e-5), ([0.0, 1.0, 0.0, -1.0], 1e-5)]:
        mu = sum(vals) / len(vals)
        var = sum((x - mu)**2 for x in vals) / len(vals)
        normed = [(x - mu) / (var + eps)**0.5 for x in vals]
        pairs.append(_qa(
            f"Apply LayerNorm (no learned params) to x = {vals}. What is the output?",
            f"μ = {mu:.2f}, σ² = {var:.4f}. "
            f"Normalized: {[round(v, 4) for v in normed]}. "
            f"Each element: (x_i - μ) / sqrt(σ² + ε)."
        ))

    return pairs



def gen_distributions(rng):
    """Gaussian, Bernoulli, Categorical, Beta — PDFs, MLE, entropy, KL formulas."""
    pairs = []

    dist_templates = [
        ("Write the PDF of the Gaussian N(μ, σ²).",
         "p(x | μ, σ²) = (1 / sqrt(2πσ²)) exp(-(x-μ)² / (2σ²)). "
         "The log-likelihood is: log p(x|μ,σ²) = -½ log(2πσ²) - (x-μ)²/(2σ²)."),
        ("What is the MLE estimate of μ and σ² for a Gaussian from data x_1,...,x_n?",
         "μ_MLE = (1/n) Σ x_i (sample mean). "
         "σ²_MLE = (1/n) Σ (x_i - μ_MLE)² (biased sample variance). "
         "The unbiased estimate uses (n-1) in the denominator, but MLE uses n."),
        ("What is the entropy of a Gaussian N(μ, σ²)?",
         "H(N(μ,σ²)) = ½ log(2πeσ²) = ½(1 + log(2πσ²)). "
         "Entropy depends only on σ², not μ. Wider distributions have higher entropy."),
        ("What is the KL divergence from N(μ₁, σ₁²) to N(μ₂, σ₂²)?",
         "KL(N₁ || N₂) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - ½. "
         "When μ₁=μ₂=0, σ₁=1, σ₂=σ: KL = log(σ) + 1/(2σ²) - ½. "
         "This form appears in VAE loss when the posterior q(z|x) is Gaussian."),
        ("What is the Bernoulli distribution and its entropy?",
         "Bernoulli(p): P(X=1)=p, P(X=0)=1-p. "
         "Entropy H = -p log p - (1-p) log(1-p). Maximum entropy at p=0.5: H = log 2 = 1 bit."),
        ("Write the log-likelihood of Bernoulli(p) for data with k successes in n trials.",
         "L(p) = k log p + (n-k) log(1-p). "
         "MLE: dL/dp = k/p - (n-k)/(1-p) = 0 → p_MLE = k/n (sample proportion)."),
        ("What is the Categorical distribution and its entropy?",
         "Categorical(π): P(X=k) = π_k for k=1,...,K, with Σ π_k = 1. "
         "Entropy H = -Σ_k π_k log π_k. Maximum H = log K at π = (1/K,...,1/K) (uniform)."),
        ("What is the reparameterization trick and why is it used?",
         "For Gaussian z ~ N(μ, σ²), write z = μ + σ * ε where ε ~ N(0,1). "
         "This makes z a deterministic function of (μ, σ, ε), so gradients can flow through z to μ and σ. "
         "Without reparameterization, ∂E[f(z)]/∂μ requires REINFORCE (high variance). "
         "Used in VAEs and any model requiring differentiable sampling."),
    ]
    pairs.extend([_qa(q, a) for q, a in dist_templates])

    # Numerical Gaussian examples
    for mu, sigma2 in [(0, 1), (2, 4), (-1, 0.25), (0, 9)]:
        sigma = sigma2**0.5
        entropy = 0.5 * math.log(2 * math.pi * math.e * sigma2)
        pairs.append(_qa(
            f"What is the entropy of N(μ={mu}, σ²={sigma2})?",
            f"H = ½ log(2πeσ²) = ½ log(2π·e·{sigma2}) = {entropy:.4f} nats. "
            f"(σ = {sigma:.4f})"
        ))

    # Bernoulli entropy examples
    for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        q = 1 - p
        h = -p * math.log(p) - q * math.log(q)
        pairs.append(_qa(
            f"What is the entropy of Bernoulli(p={p})?",
            f"H = -{p} log({p}) - {q} log({q}) = {h:.4f} nats."
        ))

    return pairs



def gen_adagrad_rmsprop(rng):
    """AdaGrad, RMSProp, AdamW, learning rate schedules, second-order intuition."""
    pairs = []

    optim_templates = [
        ("What is the AdaGrad update rule?",
         "AdaGrad accumulates squared gradients: G_t = G_{t-1} + g_t². "
         "Update: θ_t = θ_{t-1} - (η / sqrt(G_t + ε)) * g_t. "
         "Effective LR per parameter shrinks over time. Good for sparse gradients (NLP), "
         "but LR decays to 0 — never forgets old gradients."),
        ("What problem does RMSProp solve compared to AdaGrad?",
         "AdaGrad's accumulated G_t grows unboundedly, shrinking the LR to 0. "
         "RMSProp uses an exponential moving average: v_t = β v_{t-1} + (1-β) g_t². "
         "Update: θ_t = θ_{t-1} - (η / sqrt(v_t + ε)) * g_t. "
         "β (typically 0.9) lets old gradients decay, keeping LR stable."),
        ("Write the full Adam update rule.",
         "m_t = β₁ m_{t-1} + (1-β₁) g_t       (first moment, momentum) "
         "v_t = β₂ v_{t-1} + (1-β₂) g_t²      (second moment, RMSProp) "
         "m̂_t = m_t / (1-β₁^t)                 (bias correction for warmup) "
         "v̂_t = v_t / (1-β₂^t)                 (bias correction) "
         "θ_t = θ_{t-1} - η * m̂_t / (sqrt(v̂_t) + ε). "
         "Typical: β₁=0.9, β₂=0.999, ε=1e-8, η=3e-4."),
        ("What is AdamW and how does it differ from Adam with L2 regularization?",
         "Adam with L2 adds λ||θ||² to the loss, so the gradient includes λθ. "
         "The adaptive scaling divides the weight decay term by sqrt(v̂_t), making decay smaller "
         "for parameters with large gradients — incorrect behavior. "
         "AdamW decouples weight decay: θ_t = θ_{t-1} - η*(m̂_t/(sqrt(v̂_t)+ε) + λθ_{t-1}). "
         "The λθ term is not scaled by the adaptive factor — this is the correct L2 regularization for Adam."),
        ("What is a cosine learning rate schedule?",
         "η_t = η_min + ½(η_max - η_min)(1 + cos(π * t/T)). "
         "At t=0: η = η_max. At t=T: η = η_min. "
         "The cosine curve decays smoothly, avoiding the abruptness of step decay. "
         "Often combined with linear warmup: ramp from 0 to η_max over the first w steps."),
        ("What is the intuition behind momentum in gradient descent?",
         "Standard GD: θ_t = θ_{t-1} - η g_t (forgets previous gradients). "
         "Momentum: v_t = β v_{t-1} + g_t; θ_t = θ_{t-1} - η v_t. "
         "With β=0.9, the effective gradient is a weighted sum of the last ~10 steps. "
         "This dampens oscillations across gradient directions and accelerates along consistent directions."),
        ("What is gradient clipping and why is it used?",
         "If ||g|| > clip_value, rescale: g ← g * (clip_value / ||g||). "
         "Prevents exploding gradients (large gradients → large updates → loss diverges). "
         "Common in RNNs and transformers. clip_value=1.0 is standard for LLM pretraining. "
         "Important: clip the GLOBAL norm across all parameters, not per-parameter."),
        ("What is the second-order Taylor expansion used in Newton's method for optimization?",
         "f(θ + Δ) ≈ f(θ) + g^T Δ + ½ Δ^T H Δ, where g=∇f, H=∇²f (Hessian). "
         "Setting gradient to 0: H Δ = -g → Δ = -H⁻¹ g. "
         "Newton step: θ_t = θ_{t-1} - H⁻¹ g. Exact step to quadratic minimum, "
         "but H is n×n (n=# parameters), so computing H⁻¹ is O(n³) — infeasible for large models."),
    ]
    pairs.extend([_qa(q, a) for q, a in optim_templates])

    # Adam numerical trace
    for eta, b1, b2, eps_val in [(0.001, 0.9, 0.999, 1e-8), (3e-4, 0.9, 0.95, 1e-8)]:
        g = 0.5
        m1 = (1 - b1) * g
        v1 = (1 - b2) * g**2
        m1_hat = m1 / (1 - b1)
        v1_hat = v1 / (1 - b2)
        update = eta * m1_hat / (v1_hat**0.5 + eps_val)
        pairs.append(_qa(
            f"Run one Adam step: η={eta}, β₁={b1}, β₂={b2}, g₁={g}. What is Δθ₁?",
            f"m₁ = (1-{b1})·{g} = {m1:.4f}. "
            f"v₁ = (1-{b2})·{g}² = {v1:.6f}. "
            f"m̂₁ = {m1:.4f}/(1-{b1}) = {m1_hat:.4f}. "
            f"v̂₁ = {v1:.6f}/(1-{b2}) = {v1_hat:.6f}. "
            f"Δθ = {eta}·{m1_hat:.4f}/sqrt({v1_hat:.6f}) = {update:.6f}."
        ))

    # LR schedule examples
    for step, total, eta_max, eta_min in [(0, 1000, 3e-4, 3e-5), (500, 1000, 3e-4, 3e-5), (1000, 1000, 3e-4, 3e-5)]:
        coeff = 0.5 * (1 + math.cos(math.pi * step / total))
        lr = eta_min + coeff * (eta_max - eta_min)
        pairs.append(_qa(
            f"Cosine LR schedule: η_max={eta_max}, η_min={eta_min}, T={total}. What is η at step {step}?",
            f"coeff = ½(1 + cos(π·{step}/{total})) = {coeff:.4f}. "
            f"η = {eta_min} + {coeff:.4f}·({eta_max}-{eta_min}) = {lr:.2e}."
        ))

    return pairs



def build_pairs(seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    all_pairs = (
        gen_matrix_gradient(rng)
        + gen_probability(rng)
        + gen_loss_functions(rng)
        + gen_backprop(rng)
        + gen_optimisation(rng)
        + gen_information_theory(rng)
        + gen_numerical_la(rng)
        + gen_attention_math(rng)
        + gen_normalization_math(rng)
        + gen_distributions(rng)
        + gen_adagrad_rmsprop(rng)
    )
    rng.shuffle(all_pairs)
    return all_pairs


def main():
    count_only = "--count" in sys.argv
    check_only = "--check" in sys.argv

    print("Generating ML/DL math Q&A pairs...")
    pairs = build_pairs()
    print(f"  {len(pairs):,} pairs generated")

    if count_only:
        print(f"  Total pairs: {len(pairs)}")
        return

    if check_only:
        sample = random.sample(pairs, min(20, len(pairs)))
        for p in sample:
            print(p.rstrip())
            print("---")
        total_toks = sum(len(tokenise(p)) for p in pairs)
        print(f"\nTotal tokens (estimate): {total_toks:,}")
        return

    print("Tokenising...")
    all_tokens: list[int] = []
    for p in pairs:
        all_tokens.extend(tokenise(p))

    print(f"  {len(all_tokens):,} tokens total")
    out_path = "ml_math.bin"
    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(out_path)
    print(f"  Saved → {out_path}  ({arr.nbytes / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
