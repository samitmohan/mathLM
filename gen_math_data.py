"""
Generate synthetic math Q&A data and save as math_qa.bin (int32 tokens, same format as train.bin).

Usage:
    python gen_math_data.py          # generates math_qa.bin (~2M tokens)
    python gen_math_data.py --check  # print 20 random samples and token count, no write

Categories:
    - Power rule  (core; what the pretrain run was failing at)
    - Coefficients + polynomials
    - Sum / product rules (two-term)
    - Second + higher-order derivatives
    - Indefinite integrals (power rule reverse)
    - Basic trig derivatives
    - Exponential / log derivatives
    - Arithmetic (model was tested on "4+4")
"""

import sys
import math
import random
import itertools
import numpy as np
import tiktoken

# ─── tokeniser ──────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")

def tokenise(text: str) -> list[int]:
    return enc.encode(text, disallowed_special=())

# ─── math expression formatters ─────────────────────────────────────────────

def fmt_mono(coeff: int, exp: int) -> str:
    """Format coeff * x^exp as a readable string."""
    if exp == 0:
        return str(coeff)
    if coeff == 1:
        c = ""
    elif coeff == -1:
        c = "-"
    else:
        c = str(coeff)
    if exp == 1:
        return f"{c}x"
    return f"{c}x^{exp}"


def deriv_mono(coeff: int, exp: int) -> str:
    """Symbolic derivative of coeff*x^exp."""
    if exp == 0:
        return "0"
    new_coeff = coeff * exp
    new_exp   = exp - 1
    return fmt_mono(new_coeff, new_exp)


def integral_mono(coeff: int, exp: int) -> str:
    """Indefinite integral of coeff*x^exp (exp != -1)."""
    new_exp = exp + 1
    from math import gcd
    g = gcd(abs(coeff), new_exp)
    n, d = coeff // g, new_exp // g
    c_str = str(n) if d == 1 else f"({n}/{d})"
    if new_exp == 1:
        return f"{c_str}x + C"
    return f"{c_str}x^{new_exp} + C"


def fmt_poly2(c1, e1, c2, e2) -> str:
    """Format a two-term polynomial."""
    t1 = fmt_mono(c1, e1)
    t2_abs = fmt_mono(abs(c2), e2) if c2 < 0 else fmt_mono(c2, e2)
    op = " - " if c2 < 0 else " + "
    return t1 + op + t2_abs


def deriv_poly2(c1, e1, c2, e2) -> str:
    """Derivative of c1*x^e1 + c2*x^e2."""
    d1 = deriv_mono(c1, e1)
    d2 = deriv_mono(c2, e2)
    if d1 == "0" and d2 == "0":
        return "0"
    if d1 == "0":
        return d2
    if d2 == "0":
        return d1
    # check sign of second term
    # re-compute raw coefficient for sign
    raw_c2 = c2 * e2
    if raw_c2 < 0:
        return d1 + " - " + fmt_mono(abs(raw_c2), e2 - 1)
    return d1 + " + " + d2


# ─── question templates ──────────────────────────────────────────────────────
# Each template is a (question_template, answer_fn) pair.
# We instantiate them with concrete values.

DERIV_QS = [
    "What is the derivative of {expr}?",
    "Find the derivative of {expr}.",
    "Differentiate {expr} with respect to x.",
    "Compute d/dx of {expr}.",
    "What is d/dx[{expr}]?",
    "Calculate the derivative of {expr}.",
    "What does {expr} differentiate to?",
    "Find d({expr})/dx.",
]

INTEGRAL_QS = [
    "What is the integral of {expr}?",
    "Find the indefinite integral of {expr}.",
    "Integrate {expr} with respect to x.",
    "Compute ∫{expr} dx.",
    "What is ∫{expr} dx?",
]

ARITH_QS = [
    "What is {a} + {b}?",
    "Calculate {a} + {b}.",
    "What does {a} + {b} equal?",
    "What is {a} - {b}?",
    "Calculate {a} - {b}.",
    "What is {a} × {b}?",
    "What is {a} * {b}?",
    "What is {a} times {b}?",
    "What is {a} divided by {b}?",
    "What is {a} / {b}?",
]

# ─── pair generators ─────────────────────────────────────────────────────────

def _pair(q: str, a: str) -> str:
    """Return a formatted Q/A block."""
    return f"Q: {q}\nA: {a}\n\n"


def gen_power_rule(rng: random.Random) -> list[str]:
    """d/dx x^n for n in 0..80, all question templates."""
    pairs = []
    for n in range(0, 81):
        expr  = fmt_mono(1, n)
        ans   = deriv_mono(1, n)
        for tmpl in DERIV_QS:
            pairs.append(_pair(tmpl.format(expr=expr), ans))
    return pairs


def gen_coeff_power(rng: random.Random) -> list[str]:
    """d/dx (c * x^n) for varied c and n."""
    pairs = []
    coeffs = list(range(2, 13)) + [-2, -3, -5]
    exps   = list(range(1, 21))
    for c, n in itertools.product(coeffs, exps):
        expr = fmt_mono(c, n)
        ans  = deriv_mono(c, n)
        tmpl = rng.choice(DERIV_QS)
        pairs.append(_pair(tmpl.format(expr=expr), ans))
    return pairs


def gen_poly2(rng: random.Random) -> list[str]:
    """Derivative of two-term polynomials."""
    pairs = []
    for _ in range(4000):
        e1 = rng.randint(2, 10)
        e2 = rng.randint(1, e1 - 1)
        c1 = rng.randint(1, 8)
        c2_sign = rng.choice([1, -1])
        c2 = c2_sign * rng.randint(1, 8)
        expr = fmt_poly2(c1, e1, c2, e2)
        ans  = deriv_poly2(c1, e1, c2, e2)
        tmpl = rng.choice(DERIV_QS)
        pairs.append(_pair(tmpl.format(expr=expr), ans))
    return pairs


def gen_second_deriv(rng: random.Random) -> list[str]:
    """Second derivative d²/dx² of simple monomials."""
    second_qs = [
        "What is the second derivative of {expr}?",
        "Compute d²/dx²[{expr}].",
        "Find the second derivative of {expr} with respect to x.",
        "What is d²({expr})/dx²?",
    ]
    pairs = []
    for n in range(2, 30):
        for c in [1, 2, 3, 5]:
            expr = fmt_mono(c, n)
            # first deriv
            c1, e1 = c * n, n - 1
            # second deriv
            ans = deriv_mono(c1, e1) if e1 > 0 else "0"
            tmpl = rng.choice(second_qs)
            pairs.append(_pair(tmpl.format(expr=expr), ans))
    return pairs


def gen_integral_power(rng: random.Random) -> list[str]:
    """Integral of c*x^n for n >= 0."""
    pairs = []
    for n in range(0, 21):
        for c in [1, 2, 3, 4, 5]:
            expr = fmt_mono(c, n)
            ans  = integral_mono(c, n)
            tmpl = rng.choice(INTEGRAL_QS)
            pairs.append(_pair(tmpl.format(expr=expr), ans))
    return pairs


TRIG_RULES = [
    ("sin(x)",   "cos(x)"),
    ("cos(x)",   "-sin(x)"),
    ("tan(x)",   "sec^2(x)"),
    ("sec(x)",   "sec(x)tan(x)"),
    ("csc(x)",   "-csc(x)cot(x)"),
    ("cot(x)",   "-csc^2(x)"),
    ("sin^2(x)", "2sin(x)cos(x)"),
    ("cos^2(x)", "-2sin(x)cos(x)"),
]

def gen_trig(rng: random.Random) -> list[str]:
    pairs = []
    for expr, ans in TRIG_RULES:
        for tmpl in DERIV_QS:
            pairs.append(_pair(tmpl.format(expr=expr), ans))
    return pairs


EXP_LOG_RULES = [
    ("e^x",    "e^x"),
    ("ln(x)",  "1/x"),
    ("log(x)", "1/(x*ln(10))"),
    ("e^{2x}", "2e^{2x}"),
    ("e^{3x}", "3e^{3x}"),
    ("2^x",    "2^x * ln(2)"),
    ("a^x",    "a^x * ln(a)"),
]

def gen_exp_log(rng: random.Random) -> list[str]:
    pairs = []
    for expr, ans in EXP_LOG_RULES:
        for tmpl in DERIV_QS:
            pairs.append(_pair(tmpl.format(expr=expr), ans))
    return pairs


def gen_chain_rule(rng: random.Random) -> list[str]:
    """Chain rule: d/dx f(g(x)) — linear inner functions only (ax+b)."""
    chain_qs = [
        "What is the derivative of {expr}?",
        "Find d/dx of {expr}.",
        "Differentiate {expr} with respect to x.",
        "Compute the derivative of {expr}.",
    ]
    pairs = []
    # d/dx (ax+b)^n = n*a*(ax+b)^(n-1)
    for n in range(2, 12):
        for a in [2, 3, 4, 5]:
            for b in [0, 1, 2, -1]:
                inner = f"{a}x" if b == 0 else (f"{a}x + {b}" if b > 0 else f"{a}x - {abs(b)}")
                expr  = f"({inner})^{n}"
                coeff = n * a
                new_n = n - 1
                inner_str = f"({inner})" if new_n > 1 else ""
                if new_n == 0:
                    ans = str(coeff)
                elif new_n == 1:
                    ans = f"{coeff}({inner})"
                else:
                    ans = f"{coeff}({inner})^{new_n}"
                tmpl = rng.choice(chain_qs)
                pairs.append(_pair(tmpl.format(expr=expr), ans))
    # d/dx sin(ax) = a*cos(ax), d/dx cos(ax) = -a*sin(ax)
    for a in range(1, 8):
        pairs.append(_pair(f"What is the derivative of sin({a}x)?", f"{a}cos({a}x)"))
        pairs.append(_pair(f"What is the derivative of cos({a}x)?", f"-{a}sin({a}x)"))
        pairs.append(_pair(f"Find d/dx of sin({a}x).", f"{a}cos({a}x)"))
        pairs.append(_pair(f"Find d/dx of cos({a}x).", f"-{a}sin({a}x)"))
    # d/dx e^(ax) = a*e^(ax)
    for a in range(1, 10):
        pairs.append(_pair(f"What is the derivative of e^({a}x)?", f"{a}e^({a}x)"))
        pairs.append(_pair(f"Differentiate e^({a}x) with respect to x.", f"{a}e^({a}x)"))
    return pairs


def gen_product_rule(rng: random.Random) -> list[str]:
    """Product rule: d/dx [f*g] = f'g + fg'."""
    prod_qs = [
        "Find the derivative of {expr} using the product rule.",
        "Differentiate {expr}.",
        "What is d/dx[{expr}]?",
        "Compute the derivative of {expr}.",
    ]
    pairs = []
    # x^m * x^n = x^(m+n), but write it as product to teach the rule
    for m in range(1, 8):
        for n in range(1, 8):
            expr = f"x^{m} * x^{n}"
            # d/dx [x^m * x^n] = m*x^(m-1)*x^n + x^m*n*x^(n-1)
            # = m*x^(m+n-1) + n*x^(m+n-1) = (m+n)*x^(m+n-1)
            total = m + n
            ans = fmt_mono(total, total - 1)
            tmpl = rng.choice(prod_qs)
            pairs.append(_pair(tmpl.format(expr=expr), ans))
    # x^n * e^x
    for n in range(1, 6):
        expr = f"x^{n} * e^x"
        ans  = f"x^{n}*e^x + {n}x^{n-1}*e^x" if n > 1 else "x*e^x + e^x"
        pairs.append(_pair(f"What is the derivative of {expr}?", ans))
    return pairs


def gen_more_power(rng: random.Random) -> list[str]:
    """Extra power rule coverage with larger exponents and more templates."""
    extra_qs = [
        "What is the derivative of {expr}?",
        "Find d/dx[{expr}].",
        "Differentiate {expr}.",
        "Compute d/dx of {expr}.",
        "What does {expr} differentiate to?",
        "What is the slope function of {expr}?",
    ]
    pairs = []
    # high exponents
    for n in range(10, 51):
        for c in [1, 2, 3]:
            expr = fmt_mono(c, n)
            ans  = deriv_mono(c, n)
            for tmpl in extra_qs:
                pairs.append(_pair(tmpl.format(expr=expr), ans))
    # negative exponents: d/dx x^(-n) = -n*x^(-n-1)
    neg_qs = [
        "What is the derivative of 1/x^{n}?",
        "Find d/dx of x^(-{n}).",
        "Differentiate x^(-{n}) with respect to x.",
    ]
    for n in range(1, 11):
        ans = f"-{n}/x^{n+1}" if n > 0 else "0"
        for tmpl in neg_qs:
            pairs.append(_pair(tmpl.format(n=n), ans))
    # fractional: d/dx sqrt(x) = 1/(2*sqrt(x))
    pairs += [
        _pair("What is the derivative of sqrt(x)?", "1/(2*sqrt(x))"),
        _pair("Find d/dx of sqrt(x).", "1/(2*sqrt(x))"),
        _pair("Differentiate sqrt(x) with respect to x.", "1/(2*sqrt(x))"),
        _pair("What is the derivative of x^(1/2)?", "(1/2)x^(-1/2)"),
        _pair("What is the derivative of x^(1/3)?", "(1/3)x^(-2/3)"),
        _pair("What is the derivative of x^(3/2)?", "(3/2)x^(1/2)"),
    ]
    return pairs


def gen_arithmetic(rng: random.Random) -> list[str]:
    """Basic arithmetic — trimmed to avoid dominating the dataset."""
    pairs = []
    # addition (reduced range)
    for a in range(-20, 101):
        for b in range(-10, 21):
            pairs.append(_pair(rng.choice(ARITH_QS[:3]).format(a=a, b=b), str(a + b)))
    # subtraction
    for _ in range(800):
        a = rng.randint(-50, 100)
        b = rng.randint(-50, 100)
        pairs.append(_pair(rng.choice(ARITH_QS[3:5]).format(a=a, b=b), str(a - b)))
    # multiplication (small)
    for a in range(-10, 11):
        for b in range(-10, 11):
            pairs.append(_pair(rng.choice(ARITH_QS[5:8]).format(a=a, b=b), str(a * b)))
    # division (exact integer)
    for a in range(1, 30):
        for b in range(1, 10):
            if a % b == 0:
                pairs.append(_pair(rng.choice(ARITH_QS[8:]).format(a=a * b, b=b), str(a)))
    return pairs


def gen_simple_qa(rng: random.Random) -> list[str]:
    """Direct format pairs like the sample in findings.md: 'Q: ... A: ...'"""
    pairs = []
    # Make sure the exact pattern from findings is well-represented
    for n in range(0, 101):
        expr = fmt_mono(1, n)
        ans  = deriv_mono(1, n)
        pairs.append(_pair(f"What is the derivative of {expr}?", ans))
        pairs.append(_pair(f"Find d/dx of {expr}.", ans))
    # with coefficients
    for c in range(2, 11):
        for n in range(1, 21):
            expr = fmt_mono(c, n)
            ans  = deriv_mono(c, n)
            pairs.append(_pair(f"What is the derivative of {expr}?", ans))
    return pairs


# ─── main ────────────────────────────────────────────────────────────────────

def build_pairs(seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    all_pairs = (
        gen_power_rule(rng)
        + gen_coeff_power(rng)
        + gen_poly2(rng)
        + gen_second_deriv(rng)
        + gen_integral_power(rng)
        + gen_trig(rng)
        + gen_exp_log(rng)
        + gen_chain_rule(rng)
        + gen_product_rule(rng)
        + gen_more_power(rng)
        + gen_arithmetic(rng)
        + gen_simple_qa(rng)
    )
    rng.shuffle(all_pairs)
    return all_pairs


def main():
    check_only = "--check" in sys.argv

    print("Generating math Q&A pairs...")
    pairs = build_pairs()
    print(f"  {len(pairs):,} pairs generated")

    if check_only:
        sample = random.sample(pairs, 20)
        for p in sample:
            print(p.rstrip())
            print("---")
        total_toks = sum(len(tokenise(p)) for p in pairs)
        print(f"\nTotal tokens (estimate): {total_toks:,}")
        return

    # tokenise all pairs
    print("Tokenising...")
    all_tokens: list[int] = []
    for p in pairs:
        all_tokens.extend(tokenise(p))

    print(f"  {len(all_tokens):,} tokens total")

    out_path = "math_qa.bin"
    arr = np.array(all_tokens, dtype=np.int32)
    arr.tofile(out_path)
    print(f"  Saved → {out_path}  ({arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
