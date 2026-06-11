"""
Signal 3 — Semantic Attribution via Cosine Similarity.

Uses all-MiniLM-L6-v2 (384-dim) to embed the answer and the retrieved context,
then computes their cosine similarity.

Direction note
--------------
This signal returns raw cosine similarity in [0, 1].
The CGSDetector decides whether high cosine = risk or high cosine = grounding
based on the calibrated ``direction`` attribute.
"""

from typing import List
import numpy as np

from .base import BaseSignal


class CosineSignal(BaseSignal):
    """Cosine similarity between answer embedding and context embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy load

    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------ #
    def score(self, question: str, answer: str, context: str) -> float:
        """
        Returns cosine similarity between answer and context in [0, 1].
        question is accepted but not used (kept for API consistency).
        """
        self._load()
        embs = self._model.encode(
            [str(answer), str(context)],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        a, b = embs[0], embs[1]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-10:
            return 0.0
        cosine = float(np.dot(a, b) / norm)
        return max(0.0, min(1.0, cosine))

    # ------------------------------------------------------------------ #
    def score_batch(
        self,
        questions: List[str],
        answers:   List[str],
        contexts:  List[str],
    ) -> List[float]:
        """
        True batch: encode all answers and contexts in two forward passes,
        then compute pairwise cosine. Much faster than looping.
        """
        self._load()
        all_answers  = [str(a) for a in answers]
        all_contexts = [str(c) for c in contexts]

        emb_ans = self._model.encode(all_answers,  convert_to_numpy=True, show_progress_bar=False)
        emb_ctx = self._model.encode(all_contexts, convert_to_numpy=True, show_progress_bar=False)

        # Row-wise cosine
        norms_ans = np.linalg.norm(emb_ans, axis=1, keepdims=True).clip(min=1e-10)
        norms_ctx = np.linalg.norm(emb_ctx, axis=1, keepdims=True).clip(min=1e-10)
        emb_ans_n = emb_ans / norms_ans
        emb_ctx_n = emb_ctx / norms_ctx
        cosines = (emb_ans_n * emb_ctx_n).sum(axis=1)
        return [max(0.0, min(1.0, float(c))) for c in cosines]
