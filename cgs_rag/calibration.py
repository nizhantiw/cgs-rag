"""
CGS Calibration Utilities
=========================

Three independent steps:
  1. detect_signal_direction  — determine whether high cosine = risk or grounding
  2. find_optimal_threshold   — grid search over τ to maximise F1
  3. optimise_weights         — grid search over (α, β, γ) to maximise AUC

All functions operate on numpy arrays and are independent of the detector class,
making them easy to use standalone or in custom pipelines.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


# ─────────────────────────────────────────────────────────────────────────────
def detect_signal_direction(
    s3_scores: np.ndarray,
    labels:    np.ndarray,
) -> str:
    """
    Automatically detect the cosine signal direction on a labelled sample.

    Parameters
    ----------
    s3_scores : array of cosine similarities (one per sample)
    labels    : binary array (1 = hallucinated, 0 = grounded)

    Returns
    -------
    "cosine_as_risk"
        E[s3 | hallucinated] > E[s3 | grounded]
        → high cosine means the answer is *suspiciously* close to context
        → e.g. HaluEval adversarial benchmark construction

    "cosine_as_grounding"
        E[s3 | grounded] >= E[s3 | hallucinated]
        → high cosine means the answer is well-supported by context
        → e.g. real RAG pipelines, TruthfulQA, PubMedQA
    """
    labels    = np.asarray(labels).astype(int)
    s3_scores = np.asarray(s3_scores, dtype=float)

    mean_hal = s3_scores[labels == 1].mean() if (labels == 1).any() else 0.5
    mean_gnd = s3_scores[labels == 0].mean() if (labels == 0).any() else 0.5

    if mean_hal > mean_gnd:
        return "cosine_as_risk"
    return "cosine_as_grounding"


# ─────────────────────────────────────────────────────────────────────────────
def find_optimal_threshold(
    cgs_risk:     np.ndarray,
    labels:       np.ndarray,
    n_thresholds: int = 80,
) -> Tuple[float, float]:
    """
    Grid search for the decision threshold τ that maximises F1 on the
    provided labelled data.

    Parameters
    ----------
    cgs_risk     : array of CGS risk scores
    labels       : binary array (1 = hallucinated, 0 = grounded)
    n_thresholds : number of candidate thresholds to evaluate

    Returns
    -------
    (best_tau, best_f1)
    """
    labels   = np.asarray(labels).astype(int)
    cgs_risk = np.asarray(cgs_risk, dtype=float)

    best_f1, best_tau = 0.0, 0.40
    for tau in np.linspace(0.10, 0.90, n_thresholds):
        preds = (cgs_risk >= tau).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)

    return best_tau, best_f1


# ─────────────────────────────────────────────────────────────────────────────
def optimise_weights(
    s2_scores: np.ndarray,
    s3_scores: np.ndarray,
    labels:    np.ndarray,
    direction: str,
    s1_scores: Optional[np.ndarray] = None,
    step:      float = 0.05,
) -> Tuple[dict, float, float]:
    """
    Grid search over (α, β, γ) weights — constrained to α + β + γ = 1 —
    to maximise AUC-ROC.

    Parameters
    ----------
    s2_scores  : NLI entailment scores (Signal 2)
    s3_scores  : cosine similarity scores (Signal 3)
    labels     : binary array (1 = hallucinated, 0 = grounded)
    direction  : "cosine_as_risk" or "cosine_as_grounding"
    s1_scores  : token log-prob confidence (Signal 1); None = Lite mode
    step       : grid step size (smaller = finer search, slower)

    Returns
    -------
    (weights_dict, optimal_threshold, best_auc)
    where weights_dict = {"alpha": ..., "beta": ..., "gamma": ...}
    """
    labels    = np.asarray(labels).astype(int)
    s2_scores = np.asarray(s2_scores, dtype=float)
    s3_scores = np.asarray(s3_scores, dtype=float)

    # Apply signal direction to s3
    s3_risk = s3_scores if direction == "cosine_as_risk" else (1.0 - s3_scores)

    best_auc     = 0.0
    best_weights = {"alpha": 0.0, "beta": 0.15, "gamma": 0.85}
    best_tau     = 0.40

    grid = np.arange(0.0, 1.0 + step / 2, step)

    if s1_scores is not None:
        s1_scores = np.asarray(s1_scores, dtype=float)
        for alpha in grid:
            for beta in grid:
                gamma = 1.0 - alpha - beta
                if gamma < -0.001 or gamma > 1.001:
                    continue
                gamma = max(0.0, min(1.0, gamma))
                risk = alpha * (1.0 - s1_scores) + beta * (1.0 - s2_scores) + gamma * s3_risk
                try:
                    auc = roc_auc_score(labels, risk)
                except ValueError:
                    continue
                if auc > best_auc:
                    best_auc = auc
                    tau, _   = find_optimal_threshold(risk, labels)
                    best_tau = tau
                    best_weights = {
                        "alpha": round(float(alpha), 3),
                        "beta":  round(float(beta), 3),
                        "gamma": round(float(gamma), 3),
                    }
    else:
        # Lite mode: α fixed at 0, search over β + γ = 1
        for beta in grid:
            gamma = 1.0 - beta
            if gamma < 0:
                continue
            risk = beta * (1.0 - s2_scores) + gamma * s3_risk
            try:
                auc = roc_auc_score(labels, risk)
            except ValueError:
                continue
            if auc > best_auc:
                best_auc = auc
                tau, _   = find_optimal_threshold(risk, labels)
                best_tau = tau
                best_weights = {
                    "alpha": 0.0,
                    "beta":  round(float(beta), 3),
                    "gamma": round(float(gamma), 3),
                }

    return best_weights, best_tau, round(best_auc, 4)
