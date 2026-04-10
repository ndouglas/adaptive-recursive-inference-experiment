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


if __name__ == "__main__":
    probes = generate_math_probes()
    with open("data/math_probe_expanded.json", "w") as f:
        json.dump(probes, f, indent=2)
    print(f"Generated {len(probes)} math probes")

    # Print category breakdown
    from collections import Counter
    cats = Counter(p["category"] for p in probes)
    diffs = Counter(p["difficulty"] for p in probes)
    print(f"Categories: {dict(cats)}")
    print(f"Difficulties: {dict(diffs)}")
