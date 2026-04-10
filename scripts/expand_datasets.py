"""Generate expanded math probe datasets for statistical analysis.

Produces deterministic math problems with known answers across categories:
- basic_arithmetic: addition, multiplication, division
- multi_step: chained operations
- powers_roots: squares, cubes, square roots
- modular: modular arithmetic
- combinatorics: choose(n, k)
- number_theory: GCD, LCM, primes

Target: 100 problems with difficulty labels (easy/medium/hard).
"""
import json
import math
import random


def generate_math_probes(seed=42, target=100):
    random.seed(seed)
    probes = []

    # Basic arithmetic (20 problems)
    for _ in range(10):
        a, b = random.randint(10, 999), random.randint(10, 999)
        probes.append({
            "question": f"What is {a} + {b}?",
            "answer": a + b,
            "category": "basic_arithmetic",
            "difficulty": "easy",
        })
    for _ in range(10):
        a, b = random.randint(10, 99), random.randint(10, 99)
        probes.append({
            "question": f"What is {a} * {b}?",
            "answer": a * b,
            "category": "basic_arithmetic",
            "difficulty": "easy",
        })

    # Multi-step (20 problems)
    for _ in range(10):
        a, b, c = random.randint(2, 20), random.randint(2, 20), random.randint(2, 20)
        probes.append({
            "question": f"What is ({a} + {b}) * {c}?",
            "answer": (a + b) * c,
            "category": "multi_step",
            "difficulty": "medium",
        })
    for _ in range(10):
        a, b, c, d = (random.randint(2, 15) for _ in range(4))
        probes.append({
            "question": f"What is ({a} * {b}) + ({c} * {d})?",
            "answer": (a * b) + (c * d),
            "category": "multi_step",
            "difficulty": "medium",
        })

    # Powers and roots (20 problems)
    for _ in range(10):
        base = random.randint(2, 30)
        probes.append({
            "question": f"What is {base} squared?",
            "answer": base ** 2,
            "category": "powers_roots",
            "difficulty": "easy",
        })
    for _ in range(10):
        root = random.randint(2, 20)
        square = root ** 2
        probes.append({
            "question": f"What is the square root of {square}?",
            "answer": root,
            "category": "powers_roots",
            "difficulty": "medium",
        })

    # Modular arithmetic (15 problems)
    for _ in range(15):
        base = random.randint(5, 50)
        exp = random.randint(2, 4)
        mod = random.randint(3, 17)
        probes.append({
            "question": f"What is {base}^{exp} mod {mod}?",
            "answer": pow(base, exp, mod),
            "category": "modular",
            "difficulty": "hard",
        })

    # Combinatorics (10 problems)
    for _ in range(10):
        n = random.randint(4, 12)
        k = random.randint(2, min(4, n))
        probes.append({
            "question": f"How many ways can you choose {k} items from {n}?",
            "answer": math.comb(n, k),
            "category": "combinatorics",
            "difficulty": "medium",
        })

    # Number theory (15 problems)
    for _ in range(8):
        a, b = random.randint(10, 200), random.randint(10, 200)
        probes.append({
            "question": f"What is the GCD of {a} and {b}?",
            "answer": math.gcd(a, b),
            "category": "number_theory",
            "difficulty": "medium",
        })
    for _ in range(7):
        a, b = random.randint(5, 50), random.randint(5, 50)
        lcm = (a * b) // math.gcd(a, b)
        probes.append({
            "question": f"What is the LCM of {a} and {b}?",
            "answer": lcm,
            "category": "number_theory",
            "difficulty": "hard",
        })

    random.shuffle(probes)
    return probes[:target]


def generate_reasoning_probes(seed=123, target=100):
    """Generate reasoning probes across 5 categories with deterministic answers.

    Categories:
    - chained_arithmetic (20): multi-step arithmetic expressions
    - word_problem (20): parametric algebra word problems
    - conditional_logic (20): constraint satisfaction and counting
    - recursive (20): sequence and iteration problems
    - number_theory (20): GCD, LCM, divisors, totient
    """
    random.seed(seed)
    probes = []

    # ── chained_arithmetic (20) ───────────────────────────────────────────────
    # Template 1 (easy, 2 ops): (a + b) * c
    for _ in range(5):
        a = random.randint(2, 20)
        b = random.randint(2, 20)
        c = random.randint(2, 10)
        ans = (a + b) * c
        probes.append({
            "question": f"What is ({a} + {b}) * {c}?",
            "answer": ans,
            "steps": 2,
            "category": "chained_arithmetic",
            "difficulty": "easy",
        })

    # Template 2 (medium, 3 ops): a * b + c * d - e
    for _ in range(5):
        a = random.randint(2, 15)
        b = random.randint(2, 15)
        c = random.randint(2, 15)
        d = random.randint(2, 15)
        e = random.randint(1, min(a * b, c * d) - 1)
        ans = a * b + c * d - e
        probes.append({
            "question": f"What is {a} * {b} + {c} * {d} - {e}?",
            "answer": ans,
            "steps": 3,
            "category": "chained_arithmetic",
            "difficulty": "medium",
        })

    # Template 3 (medium, 4 ops): (a + b) * c - d * e
    for _ in range(5):
        a = random.randint(2, 15)
        b = random.randint(2, 15)
        c = random.randint(2, 10)
        d = random.randint(1, 5)
        e = random.randint(1, 5)
        ans = (a + b) * c - d * e
        # Ensure positive answer
        if ans <= 0:
            ans = (a + b) * c
            d, e = 0, 0
        probes.append({
            "question": f"What is ({a} + {b}) * {c} - {d} * {e}?",
            "answer": ans,
            "steps": 4,
            "category": "chained_arithmetic",
            "difficulty": "medium",
        })

    # Template 4 (hard, 5 ops): a * (b + c) - d + e * f
    for _ in range(5):
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        c = random.randint(2, 10)
        d = random.randint(1, 20)
        e = random.randint(2, 10)
        f = random.randint(2, 10)
        ans = a * (b + c) - d + e * f
        # Ensure positive answer
        if ans <= 0:
            d = 1
            ans = a * (b + c) - d + e * f
        probes.append({
            "question": f"What is {a} * ({b} + {c}) - {d} + {e} * {f}?",
            "answer": ans,
            "steps": 5,
            "category": "chained_arithmetic",
            "difficulty": "hard",
        })

    # ── word_problem (20) ─────────────────────────────────────────────────────
    # Template 1: apples / giving fraction / eating
    fractions = [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]
    for _ in range(7):
        frac_num, frac_den = random.choice(fractions)
        n = random.randint(frac_den * 4, frac_den * 20)
        # Round n to multiple of frac_den so division is exact
        n = (n // frac_den) * frac_den
        given = (n * frac_num) // frac_den
        remaining_after_give = n - given
        k = random.randint(1, max(1, remaining_after_give - 2))
        ans = remaining_after_give - k
        if ans <= 0:
            k = 1
            ans = remaining_after_give - k
        frac_str = f"{frac_num}/{frac_den}"
        probes.append({
            "question": (
                f"Alice has {n} apples. She gives {frac_str} of them to Bob, "
                f"then eats {k}. How many apples does Alice have left?"
            ),
            "answer": ans,
            "steps": 3,
            "category": "word_problem",
            "difficulty": "medium",
        })

    # Template 2: store change
    for _ in range(7):
        p = random.randint(2, 20)
        q = random.randint(2, 10)
        total = p * q
        # bill must be a round number greater than total
        bill = (total // 10 + 1) * 10 + random.choice([0, 10, 20])
        ans = bill - total
        probes.append({
            "question": (
                f"A store sells items for ${p} each. "
                f"Someone buys {q} and pays with ${bill}. "
                f"How much change do they receive?"
            ),
            "answer": ans,
            "steps": 2,
            "category": "word_problem",
            "difficulty": "easy",
        })

    # Template 3: train distance (two legs)
    for _ in range(6):
        s1 = random.choice([30, 40, 50, 60, 70, 80])
        t1 = random.randint(1, 4)
        s2 = random.choice([20, 30, 40, 50, 60])
        t2 = random.randint(1, 4)
        ans = s1 * t1 + s2 * t2
        probes.append({
            "question": (
                f"A train travels {s1} mph for {t1} hours, "
                f"then {s2} mph for {t2} hours. "
                f"How many total miles does it travel?"
            ),
            "answer": ans,
            "steps": 3,
            "category": "word_problem",
            "difficulty": "medium",
        })

    # ── conditional_logic (20) ────────────────────────────────────────────────
    # Template 1: largest integer less than n divisible by d
    for _ in range(7):
        d = random.randint(3, 12)
        n = random.randint(d * 3, d * 20)
        # Ensure n is not itself divisible by d (so the answer isn't trivially n)
        while n % d == 0:
            n += 1
        ans = n - (n % d)
        probes.append({
            "question": (
                f"What is the largest integer less than {n} "
                f"that is divisible by {d}?"
            ),
            "answer": ans,
            "steps": 2,
            "category": "conditional_logic",
            "difficulty": "easy",
        })

    # Template 2: count divisible by a but not b in [1..n]
    for _ in range(7):
        a = random.randint(2, 8)
        b = random.randint(2, 8)
        while b == a or math.gcd(a, b) == a or math.gcd(a, b) == b:
            b = random.randint(2, 8)
        n = random.randint(30, 100)
        # Count integers divisible by a but not b
        count = sum(1 for x in range(1, n + 1) if x % a == 0 and x % b != 0)
        probes.append({
            "question": (
                f"How many integers from 1 to {n} "
                f"are divisible by {a} but not by {b}?"
            ),
            "answer": count,
            "steps": 3,
            "category": "conditional_logic",
            "difficulty": "medium",
        })

    # Template 3: count divisible by a or b via inclusion-exclusion
    for _ in range(6):
        a = random.randint(2, 7)
        b = random.randint(2, 7)
        while b == a:
            b = random.randint(2, 7)
        n = random.randint(30, 100)
        lcm_ab = (a * b) // math.gcd(a, b)
        count = n // a + n // b - n // lcm_ab
        probes.append({
            "question": (
                f"How many integers from 1 to {n} "
                f"are divisible by {a} or {b} (or both)?"
            ),
            "answer": count,
            "steps": 4,
            "category": "conditional_logic",
            "difficulty": "medium",
        })

    # ── recursive (20) ────────────────────────────────────────────────────────
    # Template 1: start s, double and add k, repeat n times
    for _ in range(7):
        s = random.randint(1, 10)
        k = random.randint(1, 5)
        n = random.randint(3, 6)
        val = s
        for _ in range(n):
            val = val * 2 + k
        ans = val
        diff = "medium" if n <= 4 else "hard"
        probes.append({
            "question": (
                f"Start with {s}. Repeatedly double the number and add {k}. "
                f"After {n} repetitions, what is the result?"
            ),
            "answer": ans,
            "steps": n,
            "category": "recursive",
            "difficulty": diff,
        })

    # Template 2: population growth p * f^y
    for _ in range(7):
        p = random.choice([2, 3, 5, 10, 20, 50, 100])
        f = random.randint(2, 4)
        y = random.randint(2, 5)
        ans = p * (f ** y)
        diff = "easy" if y <= 2 else ("medium" if y <= 4 else "hard")
        probes.append({
            "question": (
                f"A population of {p} grows by a factor of {f} each year. "
                f"What is the population after {y} years?"
            ),
            "answer": ans,
            "steps": y,
            "category": "recursive",
            "difficulty": diff,
        })

    # Template 3: subtract d until below t — how many steps?
    for _ in range(6):
        d = random.randint(3, 15)
        t = random.randint(0, 20)
        s = random.randint(t + d * 4, t + d * 15)
        # Number of subtractions to go strictly below t
        steps = math.ceil((s - t) / d)
        # Verify
        val = s
        count = 0
        while val > t:
            val -= d
            count += 1
        ans = count
        probes.append({
            "question": (
                f"Start with {s}. Subtract {d} repeatedly. "
                f"After how many subtractions do you first go below {t}?"
            ),
            "answer": ans,
            "steps": 4,
            "category": "recursive",
            "difficulty": "hard",
        })

    # ── number_theory (20) ────────────────────────────────────────────────────
    # GCD (4)
    for _ in range(4):
        a = random.randint(10, 300)
        b = random.randint(10, 300)
        ans = math.gcd(a, b)
        probes.append({
            "question": f"What is the GCD of {a} and {b}?",
            "answer": ans,
            "steps": 3,
            "category": "number_theory",
            "difficulty": "medium",
        })

    # LCM (4)
    for _ in range(4):
        a = random.randint(5, 50)
        b = random.randint(5, 50)
        ans = (a * b) // math.gcd(a, b)
        probes.append({
            "question": f"What is the LCM of {a} and {b}?",
            "answer": ans,
            "steps": 3,
            "category": "number_theory",
            "difficulty": "medium",
        })

    # Number of divisors (4)
    for _ in range(4):
        n = random.randint(10, 200)
        ans = sum(1 for i in range(1, n + 1) if n % i == 0)
        probes.append({
            "question": f"How many positive divisors does {n} have?",
            "answer": ans,
            "steps": 4,
            "category": "number_theory",
            "difficulty": "hard",
        })

    # Sum of divisors (4)
    for _ in range(4):
        n = random.randint(10, 150)
        ans = sum(i for i in range(1, n + 1) if n % i == 0)
        probes.append({
            "question": f"What is the sum of all positive divisors of {n}?",
            "answer": ans,
            "steps": 4,
            "category": "number_theory",
            "difficulty": "hard",
        })

    # Euler's totient (4)
    for _ in range(4):
        n = random.randint(10, 100)
        ans = sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1)
        probes.append({
            "question": (
                f"What is Euler's totient of {n}? "
                f"(How many integers from 1 to {n} are coprime to {n}?)"
            ),
            "answer": ans,
            "steps": 4,
            "category": "number_theory",
            "difficulty": "hard",
        })

    random.shuffle(probes)
    return probes[:target]


if __name__ == "__main__":
    from collections import Counter

    probes = generate_math_probes()
    with open("data/math_probe_expanded.json", "w") as f:
        json.dump(probes, f, indent=2)
    print(f"Generated {len(probes)} math probes")
    cats = Counter(p["category"] for p in probes)
    diffs = Counter(p["difficulty"] for p in probes)
    print(f"Categories: {dict(cats)}")
    print(f"Difficulties: {dict(diffs)}")

    reasoning_probes = generate_reasoning_probes()
    with open("data/reasoning_probe_expanded.json", "w") as f:
        json.dump(reasoning_probes, f, indent=2)
    print(f"\nGenerated {len(reasoning_probes)} reasoning probes")
    rcats = Counter(p["category"] for p in reasoning_probes)
    rdiffs = Counter(p["difficulty"] for p in reasoning_probes)
    print(f"Categories: {dict(rcats)}")
    print(f"Difficulties: {dict(rdiffs)}")
