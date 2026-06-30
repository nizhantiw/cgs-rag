"""
cgs-rag  v0.2.0
===============
Composite Grounding Score — multi-signal hallucination detection for
production RAG systems.

Based on the thesis:
  "Composite Grounding Score (CGS): A Multi-Signal Framework for
   Hallucination Detection in Production RAG Systems"
  Nishant Kumar, IIT Patna M.Tech AI & DSE, 2026

Quick start — whole-answer scoring
------------------------------------
    from cgs_rag import CGSDetector

    detector = CGSDetector()
    result = detector.score(
        question = "What is the capital of France?",
        answer   = "Berlin",
        context  = "France is a country in Western Europe. Its capital is Paris."
    )
    print(result.risk_score)          # e.g. 0.782
    print(result.is_hallucination)    # True
    print(result.explain())

NEW in v0.2.0 — Atomic claim-level scoring
--------------------------------------------
    result = detector.score_claims(
        question = "What is d_model and how many attention heads?",
        answer   = "The model uses d_model=512 with 12 attention heads.",
        chunks   = [                                 # individual retrieved chunks
            "We use d_model = 512 in our base model.",
            "We employ h = 8 parallel attention heads.",
        ],
    )
    print(result.risk_score)           # 0.91  (worst claim drives the score)
    print(result.hallucinated_count)   # 1
    for c in result.claims:
        print(c.verdict, c.text)
    # GROUNDED    The model uses d_model=512
    # HALLUCINATED  12 attention heads
    print(result.explain())            # full breakdown

Domain calibration
------------------
    import pandas as pd
    val_df = pd.read_csv("my_labelled_rag_data.csv")
    # columns needed: question, answer, context, label (1=hallucinated, 0=grounded)

    detector.calibrate(val_df)
    # Automatically detects signal direction, optimises weights & threshold.
"""

from .detector import CGSDetector
from .result   import CGSResult, CGSClaimResult, CGSAtomicResult

__version__ = "0.2.0"
__author__  = "Nishant Kumar"
__all__     = ["CGSDetector", "CGSResult", "CGSClaimResult", "CGSAtomicResult"]
