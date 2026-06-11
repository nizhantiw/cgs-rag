"""Abstract base class for all CGS signals."""

from abc import ABC, abstractmethod
from typing import List


class BaseSignal(ABC):
    """
    Every signal must implement:
      - is_available  (property) : True if the signal's dependencies are installed/reachable
      - score()                  : single sample → float in [0, 1]
      - score_batch()            : list of samples → list of floats
    """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this signal can be computed (deps installed, service up)."""

    @abstractmethod
    def score(self, question: str, answer: str, context: str) -> float:
        """
        Compute a raw signal value for one QA sample.

        Returns
        -------
        float in [0, 1].
        The *direction* of the signal (whether high = hallucinated or high = grounded)
        is handled by CGSDetector, not here.
        """

    def score_batch(
        self,
        questions: List[str],
        answers:   List[str],
        contexts:  List[str],
    ) -> List[float]:
        """
        Default batch implementation: loop over samples.
        Subclasses may override for true batching (e.g. SentenceTransformer).
        """
        return [
            self.score(q, a, c)
            for q, a, c in zip(questions, answers, contexts)
        ]
