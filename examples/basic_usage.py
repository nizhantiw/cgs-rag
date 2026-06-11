"""
CGS basic usage — score RAG responses without any setup.

Install first:
    cd cgs_framework
    pip install -e .

Run:
    python examples/basic_usage.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cgs_rag import CGSDetector

# ── Initialise ────────────────────────────────────────────────────────────────
# "auto" mode: uses S1+S2+S3 if Ollama is running, else falls back to S2+S3 Lite
detector = CGSDetector()
print(f"\nDetector: {detector}")
print()

# ── Example 1: clearly hallucinated answer ────────────────────────────────────
print("=" * 60)
print("EXAMPLE 1: hallucinated (wrong entity)")
print("=" * 60)
r1 = detector.score(
    question = "What is the capital of France?",
    answer   = "Berlin, Germany's capital city, is known for its art scene.",
    context  = (
        "France is a country in Western Europe. Its capital city is Paris, "
        "known as the City of Light. Paris has a population of over 2 million."
    ),
)
print(r1.explain())
print()

# ── Example 2: correct, grounded answer ──────────────────────────────────────
print("=" * 60)
print("EXAMPLE 2: grounded (correct)")
print("=" * 60)
r2 = detector.score(
    question = "What is the capital of France?",
    answer   = "Paris is the capital of France.",
    context  = (
        "France is a country in Western Europe. Its capital city is Paris, "
        "known as the City of Light. Paris has a population of over 2 million."
    ),
)
print(r2.explain())
print()

# ── Example 3: medical RAG answer ────────────────────────────────────────────
print("=" * 60)
print("EXAMPLE 3: medical RAG — hallucinated drug name")
print("=" * 60)
r3 = detector.score(
    question = "What drug is used to treat Type 2 diabetes as a first-line therapy?",
    answer   = "Insulin is the standard first-line treatment for Type 2 diabetes.",
    context  = (
        "Metformin is the recommended first-line pharmacological treatment for "
        "Type 2 diabetes mellitus, according to major clinical guidelines. It "
        "reduces hepatic glucose production and improves insulin sensitivity."
    ),
)
print(r3.explain())
print()

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"{'Example':<15} {'Risk':>6}  {'Hallucination':>13}  S2_NLI  S3_Cosine")
print("-" * 60)
for label, r in [("Example 1", r1), ("Example 2", r2), ("Example 3", r3)]:
    print(f"{label:<15} {r.risk_score:>6.3f}  {str(r.is_hallucination):>13}  "
          f"{r.signals['s2_nli']:>6.3f}  {r.signals['s3_cosine']:>9.3f}")
print("=" * 60)
