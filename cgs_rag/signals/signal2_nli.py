"""
Signal 2 — NLI Faithfulness Score.

Uses cross-encoder/nli-deberta-v3-small to compute the probability that the
retrieved context *entails* the generated answer.

Hypothesis template (from thesis):
    "The answer to '{question}' is: {answer}."

Returns the ENTAILMENT probability in [0, 1].
High score = answer is entailed by context = grounded.
Low score  = answer contradicts or is unrelated to context = suspicious.
"""

from typing import List

from .base import BaseSignal

# DeBERTa NLI label sets differ by model variant — handle all possibilities.
_ENTAIL_LABELS = {"entailment", "entail", "label_2", "yes"}
_CONTRA_LABELS = {"contradiction", "contradict", "label_0", "no"}


class NLISignal(BaseSignal):
    """NLI entailment probability between context (premise) and answer (hypothesis)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self._pipe = None  # lazy load

    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,          # return all label scores
                truncation=True,
                max_length=512,
            )

    # ------------------------------------------------------------------ #
    def score(self, question: str, answer: str, context: str) -> float:
        """
        Returns entailment probability in [0, 1].
        Higher = answer is more faithfully supported by context.
        """
        self._load()
        premise    = str(context)
        hypothesis = f"The answer to '{question}' is: {answer}."

        raw = self._pipe(
            {"text": premise, "text_pair": hypothesis}
        )
        # raw is a list of dicts: [{"label": "ENTAILMENT", "score": 0.92}, ...]
        return self._extract_entailment(raw)

    # ------------------------------------------------------------------ #
    def _extract_entailment(self, raw) -> float:
        """
        Robustly extract entailment probability from the pipeline output,
        regardless of how the model labels its classes.
        """
        if not raw:
            return 0.5

        # Normalise to a list if wrapped in an extra list
        items = raw[0] if (isinstance(raw[0], list)) else raw

        label_score = {}
        for item in items:
            label_score[item["label"].lower()] = float(item["score"])

        # Try to find entailment label directly
        for lbl in _ENTAIL_LABELS:
            if lbl in label_score:
                return label_score[lbl]

        # Fallback: 1 - contradiction_score (works when model uses 3 classes)
        for lbl in _CONTRA_LABELS:
            if lbl in label_score:
                neutral_score = sum(
                    v for k, v in label_score.items()
                    if k not in _CONTRA_LABELS
                )
                return max(0.0, min(1.0, neutral_score))

        # Last resort: assume highest-scoring label is entailment
        return max(label_score.values()) if label_score else 0.5

    # ------------------------------------------------------------------ #
    def score_batch(
        self,
        questions: List[str],
        answers:   List[str],
        contexts:  List[str],
    ) -> List[float]:
        """Batch using transformers pipeline (handles internal batching)."""
        self._load()
        inputs = [
            {"text": str(c), "text_pair": f"The answer to '{q}' is: {a}."}
            for q, a, c in zip(questions, answers, contexts)
        ]
        results = self._pipe(inputs)
        return [self._extract_entailment(r) for r in results]
