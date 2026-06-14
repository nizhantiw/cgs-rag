"""
Tests for the atomic claim decomposer (cgs_rag/claims.py).
Run with: pytest tests/ -v
"""

import pytest
from cgs_rag.claims import decompose, claim_count


class TestDecompose:

    def test_single_sentence_returns_one_claim(self):
        answer = "The model uses d_model=512 dimensions."
        claims = decompose(answer)
        assert len(claims) == 1
        assert "d_model=512" in claims[0]

    def test_multiple_sentences_split_correctly(self):
        answer = "The model uses d_model=512. It has 8 attention heads."
        claims = decompose(answer)
        assert len(claims) == 2

    def test_semicolon_splits_independent_clauses(self):
        answer = "d_model is 512; h is 8; dff is 2048."
        claims = decompose(answer)
        assert len(claims) == 3

    def test_numbered_list_parsed(self):
        answer = "1. d_model = 512\n2. h = 8 heads\n3. dff = 2048"
        claims = decompose(answer)
        assert len(claims) == 3

    def test_decimal_numbers_not_split(self):
        # "25.8" should not be a sentence boundary
        answer = "The BLEU score was 25.8 on EN-DE. The model achieved 41.0 on EN-FR."
        claims = decompose(answer)
        assert len(claims) == 2
        assert "25.8" in claims[0]

    def test_empty_string_returns_empty_list(self):
        assert decompose("") == []

    def test_very_short_answer_returns_original(self):
        # Shorter than MIN_WORDS but still returned as fallback
        answer = "Yes."
        claims = decompose(answer)
        assert len(claims) >= 1

    def test_no_extra_punctuation_in_claims(self):
        answer = "The paper proposes the Transformer. It uses attention."
        claims = decompose(answer)
        for c in claims:
            assert not c.endswith(".")
            assert not c.endswith(",")

    def test_claim_count_helper(self):
        answer = "The model uses d_model=512. It has 8 heads."
        assert claim_count(answer) == 2


class TestImports:
    """Ensure all public exports are importable and correctly typed."""

    def test_cgs_result_importable(self):
        from cgs_rag import CGSResult
        assert CGSResult is not None

    def test_claim_result_importable(self):
        from cgs_rag import CGSClaimResult
        r = CGSClaimResult(
            text="test claim", nli=0.9, cosine=0.8,
            risk=0.1, verdict="GROUNDED", best_chunk_idx=0
        )
        assert r.verdict == "GROUNDED"

    def test_atomic_result_importable(self):
        from cgs_rag import CGSClaimResult, CGSAtomicResult
        c = CGSClaimResult(text="claim", nli=0.9, cosine=0.8, risk=0.1, verdict="GROUNDED")
        r = CGSAtomicResult(
            risk_score=0.1, is_hallucination=False, threshold=0.4,
            mode="lite", direction="cosine_as_grounding", claims=[c]
        )
        assert r.grounded_count == 1
        assert r.hallucinated_count == 0

    def test_atomic_result_to_dict(self):
        from cgs_rag import CGSClaimResult, CGSAtomicResult
        c = CGSClaimResult(text="claim", nli=0.9, cosine=0.8, risk=0.1, verdict="GROUNDED")
        r = CGSAtomicResult(
            risk_score=0.1, is_hallucination=False, threshold=0.4,
            mode="lite", direction="cosine_as_grounding", claims=[c]
        )
        d = r.to_dict()
        assert "claims" in d
        assert d["total_claims"] == 1
        assert d["grounded_count"] == 1

    def test_score_claims_method_exists(self):
        from cgs_rag import CGSDetector
        assert hasattr(CGSDetector, "score_claims")

    def test_version_is_020(self):
        import cgs_rag
        assert cgs_rag.__version__ == "0.2.0"
