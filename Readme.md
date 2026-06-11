# cgs-rag

**Composite Grounding Score — multi-signal hallucination detection for production RAG systems.**

CGS fuses three lightweight signals into a single calibrated risk score that tells you, at inference time and without ground-truth answers, whether an LLM response is faithfully grounded in the retrieved context.

| Signal | Model | What it measures |
|--------|-------|-----------------|
| S1 — Token confidence | Llama 3.2 (Ollama) | How certain the LLM was while generating the answer |
| S2 — NLI faithfulness | DeBERTa-v3-small | Whether the context *entails* the answer |
| S3 — Semantic attribution | all-MiniLM-L6-v2 | How semantically close the answer is to the context |

Validated AUC-ROC scores across three datasets:

| Dataset | CGS AUC | RAGAS AUC | ΔAUC |
|---------|---------|-----------|------|
| HaluEval QA (n=2,000) | **0.7539** | 0.5626 | +0.2813 |
| TruthfulQA (n=6,209) | **0.8313** | — | — |
| PubMedQA RAG (n=100) | **0.9720** | 0.4524 | +0.5196 |

CGS runs in **under 0.6 sec/query on CPU** with no LLM calls required (Lite mode). RAGAS on the same 100 PubMedQA queries required 60–90 minutes.

---

## Installation

```bash
git clone https://github.com/nishant-k-marmeto/cgs-rag
cd cgs-rag/cgs_framework
pip install -e .
```

**Optional — for Full mode (Signal 1):**
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
ollama serve
```

If Ollama is not running, the detector automatically falls back to Lite mode (S2+S3). No errors, no config changes needed.

---

## Quick Start

```python
from cgs_rag import CGSDetector

detector = CGSDetector()   # auto: Lite if no Ollama, Full if Ollama is running

result = detector.score(
    question = "What is the capital of France?",
    answer   = "Berlin, Germany's capital, is known for its art scene.",
    context  = "France is a country in Western Europe. Its capital is Paris."
)

print(result.risk_score)        # 0.742
print(result.is_hallucination)  # True
print(result.explain())
```

Output:
```
CGS Risk Score : 0.742  →  HALLUCINATION ⚠️
Threshold      : 0.40  |  Mode: lite  |  Direction: cosine_as_grounding

Signal breakdown:
  S2  NLI faithfulness   : 0.000  (contradicted)
  S3  Cosine similarity  : 0.303  (semantically distant from context)

Interpretation:
  → answer is semantically distant from the retrieved context
  → NLI model does not support the answer given the context
```

---

## Modes

| Mode | Signals | Latency | AUC (HaluEval) | When to use |
|------|---------|---------|----------------|-------------|
| `lite` | S2 + S3 | 0.3–0.6 sec | 0.7539 | Most deployments — no LLM needed |
| `full` | S1 + S2 + S3 | 1.5–4 sec | 0.6920* | When you control the local LLM |
| `auto` | Detects at runtime | — | — | Default — recommended |

> *Full mode was evaluated on a 500-sample subset; the marginal gain over Lite is +0.0036 AUC.

---

## Batch Scoring

```python
results = detector.score_batch(
    questions = ["What is X?", "Who invented Y?"],
    answers   = ["X is A.",    "Y was invented by Z."],
    contexts  = ["X is A used in...", "Y was invented by W in 1985."],
)

for r in results:
    print(r.risk_score, r.is_hallucination)
```

Uses true batch inference for S2 and S3 — significantly faster than looping for large batches.

---

## Domain Calibration

CGS ships with thesis-validated defaults (HaluEval direction, τ=0.40). For best results on your own domain, calibrate on 50–100 labelled samples from your pipeline:

```python
import pandas as pd, json
from cgs_rag import CGSDetector

# DataFrame needs: question, answer, context, label (1=hallucinated, 0=grounded)
val_df = pd.read_csv("my_rag_logs_labelled.csv")

detector = CGSDetector()
cal = detector.calibrate(val_df)
# [CGS] Direction detected: cosine_as_grounding
# [CGS] Weights: α=0.00  β=0.15  γ=0.85
# [CGS] Threshold: τ = 0.42
# [CGS] AUC: 0.9341

# Save and reload
with open("cgs_config.json", "w") as f:
    json.dump(detector.get_config(), f)

prod = CGSDetector.from_config(json.load(open("cgs_config.json")))
```

Calibration automatically:
- Detects signal direction (`cosine_as_risk` vs `cosine_as_grounding`) from your data
- Optimises weights (α, β, γ) via grid search to maximise AUC
- Finds the threshold τ that maximises F1

---

## Signal Direction

Signal 3 (cosine similarity) can point in two directions depending on how hallucinations are constructed in your domain:

| Direction | When it applies | How CGS uses S3 |
|-----------|----------------|-----------------|
| `cosine_as_grounding` | Real RAG pipelines, TruthfulQA | `s3_risk = 1 − cosine` |
| `cosine_as_risk` | HaluEval adversarial benchmark | `s3_risk = cosine` |

The default (`cosine_as_grounding`) is correct for all real RAG deployments. HaluEval is an adversarial benchmark where hallucinated answers deliberately borrow vocabulary from the context — this reverses the signal direction. **Always run `calibrate()` if you are unsure which direction applies.**

---

## API Reference

### `CGSDetector`

```python
CGSDetector(
    mode         = "auto",                            # "auto" | "lite" | "full"
    nli_model    = "cross-encoder/nli-deberta-v3-small",
    cosine_model = "all-MiniLM-L6-v2",
    ollama_model = "llama3.2",
    ollama_url   = "http://localhost:11434",
    threshold    = 0.40,
    weights      = {"alpha": 0.0, "beta": 0.15, "gamma": 0.85},
    direction    = "cosine_as_grounding",
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `.score(question, answer, context)` | `CGSResult` | Score one response |
| `.score_batch(questions, answers, contexts)` | `List[CGSResult]` | Batch scoring |
| `.calibrate(val_df, ...)` | `dict` | Calibrate to your domain |
| `.get_config()` | `dict` | Export calibrated parameters |
| `CGSDetector.from_config(config)` | `CGSDetector` | Reload from saved config |

### `CGSResult`

| Field | Type | Description |
|-------|------|-------------|
| `risk_score` | float | CGS risk in [0, 1] |
| `is_hallucination` | bool | `True` if `risk_score ≥ threshold` |
| `threshold` | float | threshold used |
| `mode` | str | `"lite"` or `"full"` |
| `signals` | dict | `{"s2_nli": ..., "s3_cosine": ..., "s1_logprob": ...}` |
| `direction` | str | signal direction used |
| `.explain()` | str | human-readable breakdown |
| `.to_dict()` | dict | serialisable for logging |

---

## Known Limitations

**Short-answer entity substitution (Type A):** CGS achieves only 55.9% detection (AUC 0.5393) on short hallucinated answers (≤4 words) that borrow context vocabulary. Both correct and incorrect short answers look similarly close to the context, giving Signal 3 minimal signal. This affects 3.7% of HaluEval hallucinations. For entity-extraction pipelines, pair CGS with an entity-linking check.

**Vocabulary-distant short answers (Type B):** Short answers with zero vocabulary overlap with the context score below random (AUC 0.3313) because Signal 3 reads low cosine as "grounded". This affects 2.3% of HaluEval hallucinations. These are hallucinations generated from parametric LLM memory rather than the retrieved passage.

**NLI weakness on short answers:** DeBERTa assigns low entailment probability to short, precise answers even when correct (e.g., `s2 = 0.31` for "Paris" given a Paris-context paragraph). The composite mitigates this, but NLI-only detectors produce many false positives on single-word answers.

**Signal 1 requires local model access:** Token log-probabilities are unavailable from black-box LLM APIs (GPT-4, Claude). Use Lite mode for API-based deployments.

**Static threshold:** τ is optimised offline. For production systems with distribution shift, re-run `calibrate()` periodically on fresh labelled samples.

---

## Reproducibility

All experiments use publicly available datasets and open-source models. No proprietary data was used at any stage.

| Dataset | Source |
|---------|--------|
| HaluEval QA | `pminervini/HaluEval` (HuggingFace) |
| TruthfulQA | `truthfulqa/truthful_qa` (HuggingFace) |
| PubMedQA | `qiaojin/PubMedQA` (HuggingFace) |

---

## Citation

If you use CGS in your research, please cite:

```bibtex
@mastersthesis{kumar2026cgs,
  author  = {Nishant Kumar},
  title   = {Composite Grounding Score: A Multi-Signal Framework for
             Hallucination Detection in Production {RAG} Systems},
  school  = {Indian Institute of Technology Patna},
  year    = {2026},
  program = {M.Tech in AI \& Data Science Engineering},
}
```

---

## License

MIT License — free to use, modify, and distribute.
