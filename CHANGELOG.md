# Changelog

All notable changes to `cgs-rag` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-06-14

### Added
- **`CGSDetector.score_claims(question, answer, chunks)`** — atomic claim-level
  faithfulness scoring. Decomposes the LLM answer into atomic claims using
  rule-based NLP, verifies each claim against its best-matching retrieved chunk
  via cosine similarity + NLI, and aggregates with worst-claim strategy.
  Motivated by FaithBench (NAACL 2025) finding that sentence-level granularity
  outperforms full-answer scoring for hallucination detection.

- **`cgs_rag/claims.py`** — `decompose(answer) -> List[str]` splits answers
  into atomic verifiable claims using sentence-boundary regex, semicolon
  splitting, and newline/list-marker handling. Zero new dependencies.

- **`CGSClaimResult`** dataclass — per-claim result: `text`, `nli`, `cosine`,
  `risk`, `verdict` ("GROUNDED" | "HALLUCINATED"), `best_chunk_idx`.

- **`CGSAtomicResult`** dataclass — aggregate result from `score_claims()`:
  `risk_score` (worst-claim), `is_hallucination`, `claims` list,
  `grounded_count`, `hallucinated_count`, `explain()`, `to_dict()`.

### Why this fixes the Category A failures
In whole-answer scoring (v0.1.x), correct facts in the answer raise the
aggregate NLI/cosine and mask a single hallucinated fact. With claim-level
decomposition, each fact is scored independently — one wrong fact produces
worst-claim risk ≈ 0.91 even when 2/3 claims are correct.

### Backward compatibility
`CGSDetector.score()` and `CGSDetector.score_batch()` are unchanged.
All existing code continues to work without modification.

---

## [0.1.0] — 2026-05-01

### Added
- Initial release of `cgs-rag`.
- `CGSDetector` with three scoring modes: `auto`, `lite`, `full`.
- Signal 1 (S1): token log-probability via local Ollama.
- Signal 2 (S2): NLI faithfulness via `cross-encoder/nli-deberta-v3-small`.
- Signal 3 (S3): cosine similarity via `all-MiniLM-L6-v2`.
- `CGSResult` dataclass with `explain()` and `to_dict()`.
- Domain calibration via `CGSDetector.calibrate(val_df)`.
- Batch scoring via `CGSDetector.score_batch()`.
- Direction parameter: `cosine_as_grounding` (default RAG) vs `cosine_as_risk`
  (adversarial/HaluEval regime).
