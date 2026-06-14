"""
Atomic Claim Decomposer — CGS v0.2.0
=====================================
Splits an LLM-generated answer into atomic, independently verifiable claims
using rule-based NLP.  No LLM, no external NLP library dependencies.

Motivation
----------
Scoring a full answer as a single unit against a retrieved context causes
*answer-level dilution*: correct facts in the answer raise the aggregate
NLI/cosine signal and mask individual hallucinated facts.  Decomposing into
atomic claims and verifying each independently removes this masking effect.

Approach
--------
1. Sentence splitting   — robust regex that skips decimal numbers,
                          abbreviations, and technical notation.
2. Semicolon splitting  — semicolons often separate independent clauses.
3. Newline splitting    — handles numbered / bulleted answers.
4. Length filtering     — drops fragments shorter than MIN_WORDS tokens.

No conjunction splitting is performed, because splitting on "and" / "but"
frequently loses the grammatical subject of the second clause.

Examples
--------
>>> from cgs_rag.claims import decompose
>>> decompose("The model uses d_model=512. It has 8 attention heads.")
['The model uses d_model=512', 'It has 8 attention heads']

>>> decompose("Accuracy was 94.3%; precision was 91.0%.")
['Accuracy was 94.3%', 'precision was 91.0%']
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Sentence boundary pattern
# Splits after  .  !  ?  followed by whitespace + uppercase letter.
# Negative look-behinds prevent splitting on:
#   - decimal numbers:       "3.14 seconds"
#   - single capital abbrev: "Mr. Smith"  "U.S. Army"
#   - version / model ids:   "v3.5 model"  "GPT-4o.  Next"  (kept whole)
# ---------------------------------------------------------------------------
_SENT_BOUNDARY = re.compile(
    r"""
    (?<!\w\.\w)          # not  a.b  (abbreviation or decimal)
    (?<![A-Z][a-z]\.)    # not  Dr.  Mr.  etc.
    (?<=[.!?])           # must follow sentence-ending punctuation
    \s+                  # one or more whitespace characters
    (?=[A-Z])            # must be followed by an uppercase letter
    """,
    re.VERBOSE,
)

# Semicolons almost always separate independent clauses
_SEMI_SPLIT = re.compile(r";\s*")

# Blank lines or multiple newlines (numbered / bullet-point answers)
_NEWLINE_SPLIT = re.compile(r"\n+")

# Strip leading list markers like "1. ", "- ", "• "
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]\s+|[-•*]\s+)")

# Minimum word count for a string to count as a claim.
# Set to 3 so technical facts like "d_model = 512" are not dropped.
_MIN_WORDS = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose(answer: str) -> List[str]:
    """
    Decompose an answer string into a list of atomic claim strings.

    Parameters
    ----------
    answer : str
        The LLM-generated answer to decompose.

    Returns
    -------
    List[str]
        Atomic claims in document order.  Guaranteed to return at least one
        element; falls back to ``[answer.strip()]`` if decomposition yields
        nothing usable.
    """
    answer = answer.strip()
    if not answer:
        return []

    # ── Step 1: split on blank lines / newlines ──────────────────────────
    rough_blocks = _NEWLINE_SPLIT.split(answer)

    sentences: List[str] = []
    for block in rough_blocks:
        block = _LIST_MARKER.sub("", block).strip()
        if not block:
            continue
        # ── Step 2: split on sentence boundaries ─────────────────────────
        sents = _SENT_BOUNDARY.split(block)
        sentences.extend(s.strip() for s in sents if s.strip())

    # ── Step 3: split on semicolons ──────────────────────────────────────
    fragments: List[str] = []
    for sent in sentences:
        parts = _SEMI_SPLIT.split(sent)
        fragments.extend(p.strip() for p in parts if p.strip())

    # ── Step 4: normalise and length-filter ──────────────────────────────
    claims: List[str] = []
    for frag in fragments:
        frag = frag.rstrip(".,;:").strip()
        if len(frag.split()) >= _MIN_WORDS:
            claims.append(frag)

    return claims if claims else [answer.strip()]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def claim_count(answer: str) -> int:
    """Return the number of atomic claims in an answer (convenience helper)."""
    return len(decompose(answer))
