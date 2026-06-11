"""
Signal 1 — Token Log-Probability Confidence.

Queries a locally-running Ollama instance to obtain token-level log-probabilities
for the generated answer given the question and context.  Converts the geometric
mean log-prob to a confidence score in [0, 1].

High score = model is highly confident = less likely to be a hallucination.
Low score  = model is uncertain        = hallucination risk signal.

Requirements
------------
- Ollama installed and running  (https://ollama.com)
- The model pulled: ``ollama pull llama3.2``

Graceful fallback
-----------------
If Ollama is not reachable, ``is_available`` returns False and the detector
automatically falls back to Lite mode (S2+S3 only).  ``score()`` returns 0.5
(neutral) so existing weights stay valid if called directly.
"""

import json
import math
import urllib.request
import urllib.error
from typing import List

from .base import BaseSignal


class TokenLogProbSignal(BaseSignal):
    """Token log-probability confidence via Ollama."""

    def __init__(
        self,
        model_name:  str = "llama3.2",
        ollama_url:  str = "http://localhost:11434",
        timeout_sec: int = 30,
    ):
        self.model_name  = model_name
        self.ollama_url  = ollama_url.rstrip("/")
        self.timeout_sec = timeout_sec

    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        """Ping the Ollama /api/tags endpoint to check if the service is up."""
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def score(self, question: str, answer: str, context: str) -> float:
        """
        Returns a calibrated confidence score in [0, 1].

        Prompt structure:
            Context: <context>
            Question: <question>
            Answer: <answer>

        We ask Ollama to *continue* the prompt (generation mode) and collect
        the log-probs of the answer tokens, computing the geometric mean
        probability as the confidence score.
        """
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )
        payload = json.dumps({
            "model":  self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "logprobs": True,
                "temperature": 0,   # deterministic
                "num_predict": 1,   # we only need the continuation start
            },
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            logprobs: list = data.get("logprobs") or []
            if not logprobs:
                return 0.5  # logprobs not returned by this model

            # Geometric mean probability = exp(mean log-prob)
            avg_lp   = sum(logprobs) / len(logprobs)
            conf     = math.exp(avg_lp)   # in (0, 1]
            return max(0.0, min(1.0, conf))

        except (urllib.error.URLError, json.JSONDecodeError, KeyError):
            return 0.5   # graceful fallback — neutral confidence

    # ------------------------------------------------------------------ #
    def score_batch(
        self,
        questions: List[str],
        answers:   List[str],
        contexts:  List[str],
    ) -> List[float]:
        """Sequential batch (Ollama does not expose a true batch endpoint)."""
        return [
            self.score(q, a, c)
            for q, a, c in zip(questions, answers, contexts)
        ]
