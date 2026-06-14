"""
CGS Result types — v0.2.0
==========================

CGSResult       — original whole-answer scoring result (unchanged API)
CGSClaimResult  — NEW: result for a single atomic claim
CGSAtomicResult — NEW: aggregate result from claim-level decomposition scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# Original result type (unchanged — backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# NEW — Atomic claim-level result types (v0.2.0)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CGSClaimResult:
    """
    Faithfulness verdict for a single atomic claim extracted from the answer.

    Attributes
    ----------
    text           : the atomic claim string
    nli            : NLI entailment score vs best matching chunk  [0, 1]
    cosine         : cosine similarity to best matching chunk     [0, 1]
    risk           : claim-level risk score                       [0, 1]
    verdict        : "GROUNDED" | "HALLUCINATED"
    best_chunk_idx : index of the retrieved chunk used for this claim (-1 if unknown)
    """

    text:           str
    nli:            float
    cosine:         float
    risk:           float
    verdict:        str        # "GROUNDED" | "HALLUCINATED"
    best_chunk_idx: int = -1

    def to_dict(self) -> dict:
        return {
            "text":           self.text,
            "nli":            self.nli,
            "cosine":         self.cosine,
            "risk":           self.risk,
            "verdict":        self.verdict,
            "best_chunk_idx": self.best_chunk_idx,
        }

    def __repr__(self) -> str:
        return (
            f"CGSClaimResult(verdict={self.verdict!r}, "
            f"risk={self.risk:.3f}, "
            f"text={self.text[:40]!r})"
        )


@dataclass
class CGSAtomicResult:
    """
    Result of atomic claim-level decomposition scoring.

    The answer is decomposed into atomic claims, each claim is verified
    independently against the best-matching retrieved chunk, and the final
    risk score is the worst (maximum) per-claim risk.

    Attributes
    ----------
    risk_score       : float in [0, 1] — worst-claim risk (use for threshold decisions)
    is_hallucination : True if risk_score >= threshold
    threshold        : decision threshold
    mode             : "lite" or "full"
    direction        : cosine direction setting of the detector
    claims           : list of per-claim results in answer order
    aggregation      : aggregation strategy used (always "worst_claim" for now)
    """

    risk_score:       float
    is_hallucination: bool
    threshold:        float
    mode:             str
    direction:        str
    claims:           List[CGSClaimResult] = field(default_factory=list)
    aggregation:      str = "worst_claim"

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def grounded_count(self) -> int:
        """Number of claims classified as GROUNDED."""
        return sum(1 for c in self.claims if c.verdict == "GROUNDED")

    @property
    def hallucinated_count(self) -> int:
        """Number of claims classified as HALLUCINATED."""
        return sum(1 for c in self.claims if c.verdict == "HALLUCINATED")

    @property
    def worst_claim(self) -> "CGSClaimResult | None":
        """The claim with the highest risk score, or None if no claims."""
        return max(self.claims, key=lambda c: c.risk) if self.claims else None

    # ── Human-readable explanation ────────────────────────────────────────────

    def explain(self) -> str:
        """Return a full, human-readable breakdown of atomic claim scoring."""
        verdict_str = "HALLUCINATION ⚠️" if self.is_hallucination else "GROUNDED ✅"
        n = len(self.claims)

        lines = [
            f"CGS Atomic Risk Score : {self.risk_score:.3f}  →  {verdict_str}",
            f"Aggregation : {self.aggregation}  |  "
            f"{self.grounded_count}/{n} claims grounded",
            f"Threshold   : {self.threshold:.2f}  |  Mode: {self.mode}",
            "",
            "Claim-by-claim breakdown:",
        ]

        for i, c in enumerate(self.claims, 1):
            icon = "✅" if c.verdict == "GROUNDED" else "⚠️"
            lines.append(
                f"  [{i}] {icon} {c.verdict:<12s}  "
                f"NLI={c.nli:.3f}  cos={c.cosine:.3f}  risk={c.risk:.3f}"
            )
            lines.append(f"       \"{c.text}\"")

        if self.is_hallucination and self.worst_claim is not None:
            wc = self.worst_claim
            lines += [
                "",
                f"Highest-risk claim (risk={wc.risk:.3f}):",
                f"  \"{wc.text}\"",
                f"  NLI={wc.nli:.3f}  cosine={wc.cosine:.3f}  "
                f"chunk_idx={wc.best_chunk_idx}",
                "",
                "Interpretation:",
                "  → At least one atomic claim in the answer is not faithfully",
                "    supported by any retrieved chunk.",
            ]
        else:
            lines += [
                "",
                "Interpretation:",
                "  → All atomic claims are supported by the retrieved context.",
            ]

        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialisable dict for logging / API responses."""
        return {
            "risk_score":          self.risk_score,
            "is_hallucination":    self.is_hallucination,
            "threshold":           self.threshold,
            "mode":                self.mode,
            "direction":           self.direction,
            "aggregation":         self.aggregation,
            "grounded_count":      self.grounded_count,
            "hallucinated_count":  self.hallucinated_count,
            "total_claims":        len(self.claims),
            "claims":              [c.to_dict() for c in self.claims],
        }

    def __repr__(self) -> str:
        return (
            f"CGSAtomicResult(risk={self.risk_score:.3f}, "
            f"hal={self.is_hallucination}, "
            f"claims={len(self.claims)}, "
            f"grounded={self.grounded_count}/{len(self.claims)})"
        )
