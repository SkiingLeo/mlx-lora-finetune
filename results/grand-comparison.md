# Multi-Model Comparison: Text-to-SQL LoRA Fine-Tuning on Apple Silicon

## Setup
- **Machine**: Mac Studio M1 Ultra 64GB
- **Dataset**: gretelai/synthetic_text_to_sql (5K train, 500 valid, 1K test)
- **Task**: Schema + question → SQL query
- **Training**: 600 iters, LR 1e-5, batch 1, 16 LoRA layers, grad checkpoint
- **Evaluation**: 200 test examples, manual semantic review of every SQL output
- **Format**: Text format (`{schema}\nQ: {question}\nA: {sql}`) — no chat template

## Results Summary

| Model | Params | EXACT | SQL PASS | SQL FAIL | WRONG | **Total Correct** | SQL Rate |
|-------|--------|-------|----------|----------|-------|-------------------|----------|
| Qwen3.5-0.8B | 0.8B | 49 (24.5%) | 45 (22.5%) | 79 (39.5%) | 27 (13.5%) | **94 (47.0%)** | 86.5% |
| **Qwen3.5-2B** | 1.9B | **66 (33.0%)** | 34 (17.0%) | 90 (45.0%) | **10 (5.0%)** | **100 (50.0%)** | **95.0%** |
| Qwen3.5-4B* | 4.2B | 47 (23.5%) | 33 (16.5%) | 62 (31.0%) | 58 (29.0%) | 80 (40.0%) | 71.0% |
| Mistral-Nemo-12B | 12.2B | 29 (14.5%) | 33 (16.5%) | 52 (26.0%) | 86 (43.0%) | 62 (31.0%) | 57.0% |

*\*Qwen3.5-4B: vision weights manually stripped (text-only extraction from VLM)*

## Training Stats

| Model | Val Loss (start) | Val Loss (final) | Peak RAM | Speed (tok/sec) | Training Time |
|-------|-----------------|-------------------|----------|-----------------|---------------|
| Qwen3.5-0.8B | 1.390 | 0.617 | 3.9 GB | ~475 | ~15 min |
| Qwen3.5-2B | 1.148 | 0.624 | 5.9 GB | ~180 | ~8 min |
| Qwen3.5-4B | 1.054 | 0.596 | 11.1 GB | ~115 | ~12 min |
| Mistral-Nemo-12B | 1.070 | 0.650 | 8.2 GB | ~210 | ~8 min |

## Baseline vs Fine-Tuned (Before/After)

| Model | Baseline SQL Rate | Fine-tuned SQL Rate | Δ SQL | Baseline Exact | Fine-tuned Exact | Δ Exact |
|-------|-------------------|---------------------|-------|----------------|------------------|---------|
| Qwen3.5-0.8B | 0% | 86.5% | **+86.5pp** | 0% | 24.5% | **+24.5pp** |
| Qwen3.5-2B | 3.5% | 95.0% | **+91.5pp** | 1.5% | 33.0% | **+31.5pp** |
| Qwen3.5-4B | 8.5% | 71.0% | +62.5pp | 2.5% | 23.5% | +21.0pp |
| Mistral-Nemo-12B | 26.5% | 57.0% | +30.5pp | 7.0% | 14.5% | +7.5pp |

**The LoRA effectiveness drops with model size.** The 0.8B gains +86.5pp SQL rate; the 12B only gains +30.5pp. Larger instruction-tuned models already write some SQL (Mistral: 26.5% baseline) but resist the LoRA's format push during fine-tuning. The 2B achieves the best absolute numbers AND the largest exact match improvement (+31.5pp).

Note: baseline = same model, same prompt, zero fine-tuning. The base models mostly output direct answers ("42", "3 rows") instead of SQL queries.

### Prompt Engineering vs Fine-Tuning

Could we skip fine-tuning and just add "Only answer in SQL" to the prompt? We tested this on the 0.8B model:

| Approach | SQL Rate | Exact Match |
|----------|----------|-------------|
| Base model (no instruction) | 0% | 0% |
| Base + "Only answer in SQL" | **1.5%** | **0%** |
| **LoRA fine-tuning (600 iters)** | **86.5%** | **24.5%** |

**Prompt engineering doesn't work for small models.** The 0.8B model ignores the instruction and still outputs direct answers ("1", "400") or garbage. On simple schemas without INSERT data, it occasionally generates SQL — but on real examples with data in the prompt, it computes the answer in its head instead. Fine-tuning teaches a deep behavioral pattern that prompt engineering can't replicate at this model size.

## Key Finding: The Format Compliance Paradox

**Bigger models = worse results.** Not because they're dumber, but because they're *too smart*.

The 0.8B model obediently generates SQL 86.5% of the time — it has no choice, the LoRA pushes it to output SQL format. The 12B model, being much more capable, can actually compute the answer in its head and outputs `42` instead of `SELECT COUNT(*) FROM ...`.

### WRONG Output Analysis

| Model | WRONG Rate | Numeric Answers | Short Text | Other |
|-------|-----------|-----------------|------------|-------|
| Qwen3.5-0.8B | 13.5% | — | — | 13.5% (repetition/garbage) |
| Qwen3.5-2B | 5.0% | — | — | 5.0% |
| Qwen3.5-4B | 29.0% | ~70% | ~20% | ~10% |
| Mistral-Nemo-12B | 43.0% | 81% | 13% | 6% |

The larger models generate **direct numeric answers** instead of SQL. The 12B Mistral answered with numbers 70 out of 86 WRONG outputs.

### SQL Quality (when it does write SQL)

| Model | SQL Generated | SQL Correct | **Quality Rate** |
|-------|-------------|-------------|------------------|
| Qwen3.5-0.8B | 173 | 94 | 54.3% |
| **Qwen3.5-2B** | **190** | **100** | **52.6%** |
| Qwen3.5-4B | 142 | 80 | 56.3% |
| Mistral-Nemo-12B | 114 | 62 | 54.4% |

When models DO write SQL, quality is surprisingly similar (~53-56%). The difference is entirely about **format compliance**.

## Val Loss vs Accuracy

| Model | Final Val Loss | Total Correct % |
|-------|---------------|-----------------|
| Qwen3.5-4B | 0.596 (best) | 40.0% (3rd) |
| Qwen3.5-0.8B | 0.617 | 47.0% (2nd) |
| Qwen3.5-2B | 0.624 | **50.0% (1st)** |
| Mistral-Nemo-12B | 0.650 (worst) | 31.0% (4th) |

**Val loss is misleading.** The 4B model achieves the lowest val loss but ranks 3rd in actual accuracy. Val loss measures token-level prediction, not task-level correctness. A model that predicts `42` (the correct answer) has high perplexity on the SQL tokens but low perplexity on the "answer" tokens.

## The Sweet Spot: Qwen3.5-2B

The 2B model wins because it hits the sweet spot:
1. **Smart enough** to understand SQL patterns and write better queries
2. **Not so smart** that it tries to bypass the instruction and answer directly
3. **95% SQL rate** — almost always generates SQL (vs 57% for 12B)
4. **33% exact match** — highest character-identical accuracy
5. **50% semantic accuracy** — best overall when including equivalent SQL

## Conclusions

1. **Model size has an optimal range for LoRA fine-tuning on format tasks** — too small (limited capacity) or too large (instruction override) both hurt
2. **Val loss is not a reliable proxy for task accuracy** — always evaluate on the actual task
3. **LoRA alone can't override strong instruction-following** — the 12B model's instruct training fights against the LoRA's format push
4. **The 2B model is the Goldilocks zone**: big enough to learn SQL patterns, small enough to follow the format instruction

### Potential Fixes for Larger Models
- More LoRA layers (32+ instead of 16)
- Higher learning rate or more iterations
- Chat-template format instead of text format
- Full fine-tune instead of LoRA
- Prompt engineering: "You must output ONLY a SQL query, nothing else"
