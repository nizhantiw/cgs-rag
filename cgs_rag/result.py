"""
CGSResult — the return type for every detector.score() call.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CGSResult:
    """
    Result of a single CGS hallucination detection call.

    Attributes
    ----------
    risk_score       : float in [0, 1].  Higher = more likely hallucinated.
    is_hallucination : True if risk_score >= threshold.
    threshold        : decision threshold used for this call.
    mode             : "lite" (S2+S3) or "full" (S1+S2+S3).
    signals          : per-signal scores {"s2_nli": ..., "s3_cosine": ...}.
    direction        : "cosine_as_risk" (HaluEval/adversarial) or
                       "cosine_as_grounding" (natural RAG / TruthfulQA).
    """

    risk_score:       float
    is_hallucination: bool
    threshold:        float
    mode:             str
    signals:          Dict[str, float]
    direction:        str

    # ------------------------------------------------------------------ #
    def explain(self) -> str:
        """Return a human-readable breakdown of the score."""
        verdict = "HALLUCINATION ⚠️" if self.is_hallucination else "GROUNDED ✅"
        lines = [
            f"CGS Risk Score : {self.risk_score:.3f}  →  {verdict}",
            f"Threshold      : {self.threshold:.2f}  |  Mode: {self.mode}  |  Direction: {self.direction}",
            "",
            "Signal breakdown:",
        ]

        if "s1_logprob" in self.signals:
            s1 = self.signals["s1_logprob"]
            lines.append(f"  S1  Token confidence   : {s1:.3f}"
                         f"  ({'high confidence' if s1 > 0.7 else 'low confidence'})")

        s2 = self.signals.get("s2_nli", None)
        if s2 is not None:
            tag = "entailed" if s2 > 0.6 else ("neutral" if s2 > 0.35 else "contradicted")
            lines.append(f"  S2  NLI faithfulness   : {s2:.3f}  ({tag})")

        s3 = self.signals.get("s3_cosine", None)
        if s3 is not None:
            if self.direction == "cosine_as_risk":
                tag = "suspicious — answer borrows context vocabulary" if s3 > 0.6 else "normal"
            else:
                tag = "well-grounded" if s3 > 0.6 else "semantically distant from context"
            lines.append(f"  S3  Cosine similarity  : {s3:.3f}  ({tag})")

        lines.append("")
        lines.append("Interpretation:")

        if self.is_hallucination:
            reasons = []
            if s3 is not None:
                if self.direction == "cosine_as_risk" and s3 > 0.65:
                    reasons.append("answer vocabulary closely mirrors context (adversarial distractor pattern)")
                elif self.direction == "cosine_as_grounding" and s3 < 0.35:
                    reasons.append("answer is semantically distant from the retrieved context")
            if s2 is not None and s2 < 0.35:
                reasons.append("NLI model does not support the answer given the context")
            if reasons:
                for r in reasons:
                    lines.append(f"  → {r}")
            else:
                lines.append("  → Combined signal weight exceeds threshold.")
        else:
            lines.append("  → No strong hallucination signal detected.")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Serialisable dict for logging / storage."""
        return {
            "risk_score":       self.risk_score,
            "is_hallucination": self.is_hallucination,
            "threshold":        self.threshold,
            "mode":             self.mode,
            "direction":        self.direction,
            **self.signals,
        }

    def __repr__(self) -> str:
        return (
            f"CGSResult(risk={self.risk_score:.3f}, "
            f"hal={self.is_hallucination}, "
            f"mode={self.mode!r})"
        )
