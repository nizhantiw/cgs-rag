"""
cgs-rag
=======
Composite Grounding Score — multi-signal hallucination detection for
production RAG systems.

Based on the thesis:
  "Composite Grounding Score (CGS): A Multi-Signal Framework for
   Hallucination Detection in Production RAG Systems"
  Nishant Kumar, IIT Patna Executive M.Tech AI & DSE, 2025

Quick start
-----------
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

Domain calibration
------------------
    import pandas as pd
    val_df = pd.read_csv("my_labelled_rag_data.csv")
    # columns needed: question, answer, context, label (1=hallucinated, 0=grounded)

    detector.calibrate(val_df)
    # Automatically detects signal direction, optimises weights & threshold.
"""

from .detector import CGSDetector
from .result   import CGSResult

__version__ = "0.1.0"
__author__  = "Nishant Kumar"
__all__     = ["CGSDetector", "CGSResult"]
