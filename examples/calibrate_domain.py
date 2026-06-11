"""
CGS domain calibration — adapt weights, direction, and threshold to your data.

This example shows how to:
  1. Build a labelled validation DataFrame from your RAG pipeline logs
  2. Call detector.calibrate() to auto-tune all parameters
  3. Save the calibrated config and reload it in production

Run:
    python examples/calibrate_domain.py
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from cgs_rag import CGSDetector

# ── Build (or load) a labelled validation set ─────────────────────────────────
# In production: replace this with pd.read_csv("your_rag_logs_labelled.csv")
#
# Required columns:
#   question  — the user's question
#   answer    — the LLM's generated answer
#   context   — the retrieved passage(s)
#   label     — 1 = hallucinated, 0 = grounded  (binary int)

val_df = pd.DataFrame({
    "question": [
        "What year was the Eiffel Tower built?",
        "Who wrote Romeo and Juliet?",
        "What is the boiling point of water?",
        "Which planet is closest to the Sun?",
        "What is the speed of light?",
        "Who painted the Mona Lisa?",
    ],
    "answer": [
        "The Eiffel Tower was built in 1889.",                          # grounded
        "Romeo and Juliet was written by Charles Dickens.",             # hallucinated
        "Water boils at 100 degrees Celsius at sea level.",            # grounded
        "Venus is the closest planet to the Sun.",                     # hallucinated
        "The speed of light is approximately 300,000 km/s.",           # grounded
        "The Mona Lisa was painted by Pablo Picasso in the 1500s.",    # hallucinated
    ],
    "context": [
        "The Eiffel Tower is a wrought-iron lattice tower built in 1889 for the World's Fair.",
        "Romeo and Juliet is a tragedy written by William Shakespeare.",
        "Water boils at 100°C (212°F) at standard atmospheric pressure.",
        "Mercury is the closest planet to the Sun, followed by Venus.",
        "The speed of light in a vacuum is 299,792,458 metres per second.",
        "The Mona Lisa is a portrait by the Italian Renaissance artist Leonardo da Vinci.",
    ],
    "label": [0, 1, 0, 1, 0, 1],   # 1 = hallucinated
})

print(f"Validation set: {len(val_df)} samples "
      f"({val_df['label'].sum()} hallucinated, {(val_df['label']==0).sum()} grounded)")
print()

# ── Calibrate ─────────────────────────────────────────────────────────────────
detector = CGSDetector()
cal      = detector.calibrate(val_df, verbose=True)

print()
print("Calibrated config:")
print(json.dumps(detector.get_config(), indent=2))

# ── Save calibration to disk ──────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "cgs_config.json")
with open(config_path, "w") as f:
    json.dump(detector.get_config(), f, indent=2)
print(f"\nConfig saved to: {config_path}")

# ── Reload and use in production ──────────────────────────────────────────────
with open(config_path) as f:
    saved_config = json.load(f)

prod_detector = CGSDetector.from_config(saved_config)
print(f"\nProduction detector loaded: {prod_detector}")

# Score a new response with the calibrated detector
result = prod_detector.score(
    question = "Who wrote Romeo and Juliet?",
    answer   = "It was written by Charles Dickens.",
    context  = "Romeo and Juliet is a tragedy written by William Shakespeare.",
)
print(f"\nTest score: {result}")
print(result.explain())
