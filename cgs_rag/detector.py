"""
CGSDetector — the main public class.

Quick start
-----------
    from cgs_rag import CGSDetector

    detector = CGSDetector()                        # auto-detect mode
    result   = detector.score(q, answer, context)  # single response
    results  = detector.score_batch(qs, ans, ctxs) # batch
    detector.calibrate(val_df)                      # adapt to your domain

v0.2.0 — Atomic claim-level scoring
-------------------------------------
    result = detector.score_claims(
        question = "What is d_model?",
        answer   = "The model uses d_model=512 with 12 attention heads.",
        chunks   = ["d_model=512 in base model", "h=8 parallel attention heads"],
    )
    print(result.risk_score)          # 0.91  (worst claim: 12 heads ≠ 8)
    print(result.claims[1].verdict)   # "HALLUCINATED"
    print(result.explain())

Modes
-----
"auto"  Uses S1+S2+S3 (Full) if Ollama is reachable, else S2+S3 (Lite).
"lite"  S2 (NLI) + S3 (cosine).  No LLM required.  Recommended default.
"full"  S1 (token log-prob) + S2 + S3.  Requires local Ollama instance.

Direction
---------
"cosine_as_risk"     : high cosine → hallucinated (HaluEval adversarial regime)
"cosine_as_grounding": high cosine → grounded     (real RAG / TruthfulQA)

If you don't call calibrate(), the default direction is "cosine_as_grounding"
(the natural RAG regime), which is correct for most real-world deployments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Union

from .result import CGSResult, CGSClaimResult, CGSAtomicResult
from .signals.signal1_token import TokenLogProbSignal
from .signals.signal2_nli   import NLISignal
from .signals.signal3_cosine import CosineSignal
from .calibration import (
    detect_signal_direction,
    find_optimal_threshold,
    optimise_weights,
)

# ── Thesis-validated defaults (HaluEval test set) ────────────────────────────
_DEFAULT_WEIGHTS    = {"alpha": 0.0, "beta": 0.15, "gamma": 0.85}
_DEFAULT_THRESHOLD  = 0.40
_DEFAULT_DIRECTION  = "cosine_as_grounding"   # safe default for real RAG


class CGSDetector:
    """
    Composite Grounding Score hallucination detector.

    Parameters
    ----------
    mode         : "auto" | "lite" | "full"
    nli_model    : HuggingFace model ID for Signal 2
    cosine_model : SentenceTransformer model ID for Signal 3
    ollama_model : Ollama model name for Signal 1
    ollama_url   : URL of the local Ollama service
    threshold    : decision threshold τ (overridden by calibrate())
    weights      : {"alpha", "beta", "gamma"} weight dict
    direction    : "cosine_as_risk" or "cosine_as_grounding"
    """

    def __init__(
        self,
        mode:         str   = "auto",
        nli_model:    str   = "cross-encoder/nli-deberta-v3-small",
        cosine_model: str   = "all-MiniLM-L6-v2",
        ollama_model: str   = "llama3.2",
        ollama_url:   str   = "http://localhost:11434",
        threshold:    float = _DEFAULT_THRESHOLD,
        weights:      Optional[dict] = None,
        direction:    str   = _DEFAULT_DIRECTION,
    ):
        self.mode      = mode
        self.threshold = threshold
        self.weights   = dict(weights) if weights else dict(_DEFAULT_WEIGHTS)
        self.direction = direction
        self._calibrated = False

        # Signals are lazy-loaded on first use
        self._s1 = TokenLogProbSignal(model_name=ollama_model, ollama_url=ollama_url)
        self._s2 = NLISignal(model_name=nli_model)
        self._s3 = CosineSignal(model_name=cosine_model)

    # ── Mode resolution ──────────────────────────────────────────────────────
    @property
    def active_mode(self) -> str:
        """Resolves "auto" to "full" or "lite" based on Ollama availability."""
        if self.mode == "auto":
            return "full" if self._s1.is_available else "lite"
        return self.mode

    # ── Internal risk computation ─────────────────────────────────────────────
    def _compute_risk(self, s1: float, s2: float, s3: float) -> float:
        s3_risk = s3 if self.direction == "cosine_as_risk" else (1.0 - s3)
        a = self.weights.get("alpha", 0.0)
        b = self.weights.get("beta",  0.15)
        g = self.weights.get("gamma", 0.85)
        return float(np.clip(a * (1.0 - s1) + b * (1.0 - s2) + g * s3_risk, 0.0, 1.0))

    # ── Public API ────────────────────────────────────────────────────────────
    def score(self, question: str, answer: str, context: str) -> CGSResult:
        """
        Score a single RAG response.

        Parameters
        ----------
        question : the user's question
        answer   : the RAG system's generated answer
        context  : the retrieved context passage(s)

        Returns
        -------
        CGSResult with risk_score, is_hallucination, signals, and explain()
        """
        s2 = self._s2.score(question, answer, context)
        s3 = self._s3.score(question, answer, context)
        s1 = self._s1.score(question, answer, context) if self.active_mode == "full" else 0.5

        risk    = self._compute_risk(s1, s2, s3)
        signals = {
            "s2_nli":     round(s2, 4),
            "s3_cosine":  round(s3, 4),
        }
        if self.active_mode == "full":
            signals["s1_logprob"] = round(s1, 4)

        return CGSResult(
            risk_score       = round(risk, 4),
            is_hallucination = risk >= self.threshold,
            threshold        = self.threshold,
            mode             = self.active_mode,
            signals          = signals,
            direction        = self.direction,
        )

    # ── Atomic claim-level scoring (v0.2.0) ──────────────────────────────────
    def score_claims(
        self,
        question: str,
        answer:   str,
        chunks:   List[str],
    ) -> CGSAtomicResult:
        """
        Atomic claim-level faithfulness scoring  (CGS v0.2.0).

        Decomposes *answer* into atomic claims, finds the best-matching
        retrieved chunk for each claim via cosine similarity, then runs NLI
        to verify each claim against its best chunk independently.

        Aggregation: **worst-claim** — the overall risk score is the maximum
        per-claim risk, so a single hallucinated fact in an otherwise correct
        answer is correctly flagged.

        Parameters
        ----------
        question : str
            The user's original question.
        answer : str
            The RAG system's generated answer to decompose.
        chunks : List[str]
            Individual retrieved chunk texts (not concatenated).
            Pass the raw list from your vector store retrieval step.

        Returns
        -------
        CGSAtomicResult
            Contains per-claim details and the aggregated worst-claim risk.

        Notes
        -----
        * S1 (token log-probability) is not computed at claim level because
          it requires the full answer token sequence.  Claim-level scoring
          always uses the lite formula: risk = β*(1−NLI) + γ*(1−cosine).
        * If ``chunks`` is empty, falls back to whole-answer scoring and
          wraps the result in a single-claim CGSAtomicResult.
        * Claim cosine is always treated as a grounding signal (high cosine
          = answer claim is close to the supporting chunk) regardless of the
          detector's global ``direction`` setting.
        """
        from .claims import decompose

        # ── Fallback: no chunks supplied ────────────────────────────────────
        if not chunks:
            whole = self.score(question, answer, "")
            fallback_claim = CGSClaimResult(
                text=answer,
                nli=whole.signals.get("s2_nli", 0.0),
                cosine=whole.signals.get("s3_cosine", 0.0),
                risk=whole.risk_score,
                verdict="HALLUCINATED" if whole.is_hallucination else "GROUNDED",
                best_chunk_idx=-1,
            )
            return CGSAtomicResult(
                risk_score=whole.risk_score,
                is_hallucination=whole.is_hallucination,
                threshold=self.threshold,
                mode=self.active_mode,
                direction=self.direction,
                claims=[fallback_claim],
                aggregation="whole_answer_fallback",
            )

        # ── Step 1: decompose answer into atomic claims ──────────────────────
        claims_text = decompose(answer)

        # ── Step 2: ensure models are loaded ────────────────────────────────
        self._s3._load()
        self._s2._load()

        # ── Step 3: encode claims + chunks in batch (single forward passes) ──
        import numpy as np

        all_texts     = claims_text + [str(c) for c in chunks]
        all_embeddings = self._s3._model.encode(
            all_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        n_claims = len(claims_text)
        emb_claims = all_embeddings[:n_claims]           # shape (n_claims, dim)
        emb_chunks = all_embeddings[n_claims:]           # shape (n_chunks, dim)

        # Row-normalise for cosine similarity
        norm_c = np.linalg.norm(emb_claims, axis=1, keepdims=True).clip(min=1e-10)
        norm_k = np.linalg.norm(emb_chunks, axis=1, keepdims=True).clip(min=1e-10)
        emb_claims_n = emb_claims / norm_c
        emb_chunks_n = emb_chunks / norm_k

        # sim_matrix[i, j] = cosine(claim_i, chunk_j)  shape (n_claims, n_chunks)
        sim_matrix = emb_claims_n @ emb_chunks_n.T

        best_chunk_indices = sim_matrix.argmax(axis=1)   # (n_claims,) int
        best_cosines       = sim_matrix.max(axis=1)      # (n_claims,) float

        # ── Step 4: NLI — each claim vs its best matching chunk ──────────────
        nli_inputs = [
            {
                "text":      str(chunks[best_chunk_indices[i]]),
                "text_pair": f"The answer to '{question}' is: {claims_text[i]}.",
            }
            for i in range(n_claims)
        ]
        nli_raw    = self._s2._pipe(nli_inputs)
        nli_scores = [self._s2._extract_entailment(r) for r in nli_raw]

        # ── Step 5: per-claim risk (lite formula — cosine always as grounding)
        beta  = self.weights.get("beta",  0.15)
        gamma = self.weights.get("gamma", 0.85)

        claim_results: List[CGSClaimResult] = []
        for i in range(n_claims):
            cos = float(np.clip(best_cosines[i], 0.0, 1.0))
            nli = float(nli_scores[i])
            risk = float(np.clip(
                beta  * (1.0 - nli) +
                gamma * (1.0 - cos),
                0.0, 1.0,
            ))
            verdict = "HALLUCINATED" if risk >= self.threshold else "GROUNDED"
            claim_results.append(CGSClaimResult(
                text           = claims_text[i],
                nli            = round(nli,  4),
                cosine         = round(cos,  4),
                risk           = round(risk, 4),
                verdict        = verdict,
                best_chunk_idx = int(best_chunk_indices[i]),
            ))

        # ── Step 6: worst-claim aggregation ─────────────────────────────────
        worst_risk = max(c.risk for c in claim_results)

        return CGSAtomicResult(
            risk_score       = round(worst_risk, 4),
            is_hallucination = worst_risk >= self.threshold,
            threshold        = self.threshold,
            mode             = self.active_mode,
            direction        = self.direction,
            claims           = claim_results,
            aggregation      = "worst_claim",
        )

    # ── Batch scoring ─────────────────────────────────────────────────────────
    def score_batch(
        self,
        questions: List[str],
        answers:   List[str],
        contexts:  List[str],
    ) -> List[CGSResult]:
        """
        Score a batch of RAG responses.
        Uses true batch inference for S2 and S3 (much faster than looping).

        Returns
        -------
        List of CGSResult objects (same order as input).
        """
        n = len(questions)
        assert len(answers) == n and len(contexts) == n, \
            "questions, answers, contexts must have the same length"

        # Batch signal computation
        s2_arr = np.array(self._s2.score_batch(questions, answers, contexts))
        s3_arr = np.array(self._s3.score_batch(questions, answers, contexts))

        if self.active_mode == "full":
            s1_arr = np.array(self._s1.score_batch(questions, answers, contexts))
        else:
            s1_arr = np.full(n, 0.5)

        results = []
        for i in range(n):
            risk    = self._compute_risk(s1_arr[i], s2_arr[i], s3_arr[i])
            signals = {
                "s2_nli":    round(float(s2_arr[i]), 4),
                "s3_cosine": round(float(s3_arr[i]), 4),
            }
            if self.active_mode == "full":
                signals["s1_logprob"] = round(float(s1_arr[i]), 4)

            results.append(CGSResult(
                risk_score       = round(float(risk), 4),
                is_hallucination = float(risk) >= self.threshold,
                threshold        = self.threshold,
                mode             = self.active_mode,
                signals          = signals,
                direction        = self.direction,
            ))
        return results

    # ── Domain calibration ────────────────────────────────────────────────────
    def calibrate(
        self,
        val_df:       pd.DataFrame,
        question_col: str  = "question",
        answer_col:   str  = "answer",
        context_col:  str  = "context",
        label_col:    str  = "label",
        verbose:      bool = True,
    ) -> dict:
        """
        Calibrate CGS weights, threshold, and direction to your domain.

        Parameters
        ----------
        val_df       : labelled DataFrame.  Needs columns:
                       question, answer, context, label (1=hallucinated, 0=grounded)
        question_col : column name for questions
        answer_col   : column name for answers
        context_col  : column name for contexts
        label_col    : column name for binary labels (1=hallucinated, 0=grounded)
        verbose      : print progress and results

        Returns
        -------
        dict with {"weights", "threshold", "direction", "auc", "n_samples"}

        Side effects
        ------------
        Updates self.weights, self.threshold, self.direction, self._calibrated
        """
        if verbose:
            print(f"[CGS] Calibrating on {len(val_df):,} samples …")

        qs  = val_df[question_col].tolist()
        ans = val_df[answer_col].tolist()
        ctx = val_df[context_col].tolist()
        lbl = val_df[label_col].values.astype(int)

        if verbose: print("[CGS]   Computing S3 (cosine) …")
        s3 = np.array(self._s3.score_batch(qs, ans, ctx))

        if verbose: print("[CGS]   Computing S2 (NLI) …")
        s2 = np.array(self._s2.score_batch(qs, ans, ctx))

        s1 = None
        if self.active_mode == "full":
            if verbose: print("[CGS]   Computing S1 (token log-prob) …")
            s1 = np.array(self._s1.score_batch(qs, ans, ctx))

        # Direction auto-detection
        self.direction = detect_signal_direction(s3, lbl)
        if verbose:
            print(f"[CGS]   Direction detected: {self.direction}")
            mean_hal = s3[lbl == 1].mean()
            mean_gnd = s3[lbl == 0].mean()
            print(f"         E[s3|hal]={mean_hal:.3f}   E[s3|gnd]={mean_gnd:.3f}")

        # Weight + threshold optimisation
        self.weights, self.threshold, best_auc = optimise_weights(
            s2, s3, lbl, self.direction, s1
        )
        self._calibrated = True

        result = {
            "weights":   self.weights,
            "threshold": round(self.threshold, 3),
            "direction": self.direction,
            "auc":       best_auc,
            "n_samples": len(val_df),
        }

        if verbose:
            print(f"[CGS] Calibration complete:")
            print(f"      Weights  : α={self.weights['alpha']:.2f}  "
                  f"β={self.weights['beta']:.2f}  γ={self.weights['gamma']:.2f}")
            print(f"      Threshold: τ = {self.threshold:.2f}")
            print(f"      AUC      : {best_auc:.4f}")

        return result

    # ── Serialisation ─────────────────────────────────────────────────────────
    def get_config(self) -> dict:
        """Export calibrated configuration for storage / reproducibility."""
        return {
            "mode":        self.active_mode,
            "weights":     self.weights,
            "threshold":   self.threshold,
            "direction":   self.direction,
            "calibrated":  self._calibrated,
        }

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "CGSDetector":
        """Reconstruct a calibrated detector from a saved config dict."""
        detector = cls(
            mode      = config.get("mode", "auto"),
            weights   = config.get("weights"),
            threshold = config.get("threshold", _DEFAULT_THRESHOLD),
            direction = config.get("direction", _DEFAULT_DIRECTION),
            **kwargs,
        )
        detector._calibrated = config.get("calibrated", False)
        return detector

    # ── Repr ──────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"CGSDetector("
            f"mode={self.active_mode!r}, "
            f"τ={self.threshold:.2f}, "
            f"direction={self.direction!r}, "
            f"calibrated={self._calibrated})"
        )
