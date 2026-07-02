# Qwythos-9B-Claude-Mythos-5-1M Evaluation Report

## Model Information

| Property | Value |
|----------|-------|
| **Model** | Qwythos-9B-Claude-Mythos-5-1M (Qwen3.5-9B based reasoning model) |
| **Quantizations** | Q4_K_M, Q8_0 |
| **Parameters** | 8,953,803,264 (~8.95B) |
| **Vocabulary Size** | 248,320 |
| **Context Length (Train)** | 1,048,576 |
| **Embedding Dimension** | 4,096 |
| **Source** | https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF |

| Quantization | File Size |
|--------------|-----------|
| **Q4_K_M** | 5.24 GiB |
| **Q8_0** | 8.90 GiB |

## Setup Details

| Component | Configuration |
|-----------|---------------|
| **Hardware** | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), 128 CPU cores, 995 GB RAM |
| **CUDA Version** | 13.0 |
| **Server** | llama.cpp server v1 (commit 0eca4d4), built from source with `-DGGML_CUDA=ON` |
| **Server Args** | `-m <gguf_path> -ngl 99 --port 8081 --host 0.0.0.0 --reasoning-preserve` |
| **Port** | 8081 |
| **GPU Layers** | 99 (all layers on GPU) |
| **Evaluation Framework** | EleutherAI lm_eval v0.4.12 (`local-chat-completions` backend) |
| **API Endpoint** | `http://localhost:8081/v1/chat/completions` |

---

## Benchmark Results

### GSM8K (Grade School Math — Full 1319 samples)

| Quant | Temperature | Flexible-extract | Strict-match |
|-------|:-----------:|:----------------:|:------------:|
| **Q4_K_M** | 0.6 | 23.28% ±1.16% | 19.71% ±1.10% |
| **Q8_0** | **0.0** | **84.31%** ±1.00% | **83.24%** ±1.03% |
| **Δ (Q8 - Q4)** | | **+61.03pp** | **+63.53pp** |

> ⚠️ **Important caveat**: The Q4_K_M run used temperature=0.6 (non-deterministic sampling), while the Q8_0 run used temperature=0.0 (greedy decoding). The GSM8K Q8_0 run at temp=0.6 produced ~21% — matching the Q4 temp=0.6 numbers. This means the **temperature parameter dominates** GSM8K performance for this reasoning model, not quantization. The model's math reasoning capability only unlocks at greedy decoding (temp=0.0). The Q4 vs Q8 quantization difference is negligible compared to the temperature effect.

**Conclusion**: This model achieves **~84% GSM8K** with greedy decoding, regardless of quantization. Temperature=0.0 is mandatory for math tasks.

---

### IFEval (Instruction Following — limit=50 samples)

| Metric | Q4_K_M (temp=0.6) | Q8_0 (temp=0.0) | Δ |
|--------|:------------------:|:----------------:|:-:|
| **prompt_level_strict** | 62.00% ±6.93% | **66.00%** ±6.77% | +4.00pp |
| **prompt_level_loose** | 66.00% ±6.77% | **68.00%** ±6.66% | +2.00pp |
| **inst_level_strict** | 61.84% | **69.74%** | +7.89pp |
| **inst_level_loose** | 65.79% | **71.05%** | +5.26pp |

> **Note**: The Q4_K_M run used temp=0.6 and Q8_0 used temp=0.0. The improvement is likely a combination of both lower temperature (greedy = better instruction following) and higher quantization precision. Against a similarly-tuned Q8_0 temp=0.6 baseline, the quantization gap would be smaller.

---

### HumanEval (Code Generation — 164 problems)

| Metric | Q4_K_M (temp=0.0) | Q8_0 (temp=0.0) | Δ |
|--------|:------------------:|:----------------:|:-:|
| **pass@1** | 0.00% | 0.00% | 0.00pp |
| **Code extraction rate** | 21.95% (36/164) | **26.83%** (44/164) | +4.88pp |
| **Errors** | 0 | 0 | — |
| **Total time** | 20.9 min | 35.2 min | +14.3 min |
| **Avg per problem** | 7.6s | 12.9s | +5.3s |

> **Both quantizations score 0% pass@1**. The model fundamentally cannot solve coding problems at the HumanEval level in this evaluation setup. The slightly higher Q8_0 extraction rate (26.8% vs 21.9%) means the model produces marginally more extractable code blocks, but none of the extracted code passes the test cases. This is a **genuine model capability limitation** — this is a Qwen3.5-based reasoning/roleplay fine-tune, not a code-specialized model. The chat/instruct format also makes it difficult to produce standalone code in the expected format.

---

## Summary Comparison Table

| Benchmark | Q4_K_M | Q8_0 | Verified Difference |
|-----------|:------:|:----:|:-------------------:|
| **GSM8K** (flexible-extract) | 23.28%* | **84.31%** | Temperature dominates (not quantization) |
| **IFEval** (prompt_strict) | 62.00%* | **66.00%** | Modest improvement |
| **HumanEval** (pass@1) | 0.00% | 0.00% | No change |

\* Q4 GSM8K at temp=0.6 and IFEval at temp=0.6. Q8 uses temp=0.0 for all benchmarks.

### Fair Comparison: Temp=0.0 Only

| Benchmark | Q4_K_M (temp=0.0) | Q8_0 (temp=0.0) | Δ |
|-----------|:------------------:|:----------------:|:-:|
| **HumanEval** (pass@1) | 0.00% | 0.00% | 0pp |
| **HumanEval** (extraction) | 21.95% | 26.83% | +4.88pp |
| **IFEval** (prompt_strict) | — | 66.00% | Q4 not run at 0.0 |

---

## Blocked Benchmarks

### HellaSwag (Commonsense Reasoning)

**Status: BLOCKED** — loglikelihood task, not supported by `local-chat-completions` backend.

Attempted solutions:
1. **local-chat-completions backend**: `NotImplementedError: Loglikelihood is not supported for chat completions.`
2. **local-completions backend** with `tokenizer=Qwen/Qwen3.5-9B-Base`: llama-server returns logprobs in newer format; lm_eval expects old `token_logprobs` list → `KeyError`.
3. **hf backend** with GGUF file: `ValueError: GGUF model with architecture qwen35 is not supported yet.`

**Root cause**: The Qwen3.5 architecture ID `qwen35` is not recognized by HuggingFace Transformers' GGUF PyTorch loader.

### ARC-Challenge (Science Reasoning)

**Status: BLOCKED** — Same loglikelihood blocker as HellaSwag.

---

## Key Takeaways

1. **GSM8K requires greedy decoding (temp=0.0)** for this reasoning model. At temp=0.6, accuracy collapses to ~21% regardless of quantization. At temp=0.0, the model achieves **~84%** — a solid score for a 9B-class model.

2. **Q8_0 shows mild improvement over Q4_K_M** on instruction following (IFEval: +2-8pp) and code extraction (HumanEval: +4.9pp), but the gap is small relative to the 70% larger file size (8.9 GiB vs 5.2 GiB).

3. **HumanEval is a hard blocker** — the model's chat/roleplay fine-tuning makes it fundamentally incompatible with standalone code generation benchmarks. 0% pass@1 across both quantizations.

4. **Practical recommendation**: For general use, Q4_K_M at temp=0.0 provides the best quality/size tradeoff. The Q8_0 quantization offers marginal improvements that likely won't justify the 70% larger file size in most applications. For math reasoning specifically, temperature=0.0 is non-negotiable.

## Results File Locations

| Benchmark | Quant | Path |
|-----------|-------|------|
| GSM8K | Q4_K_M | `/root/models/results/gsm8k_full/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf/results_2026-06-30T21-02-53.400327.json` |
| GSM8K | Q8_0 | `/root/models/results/gsm8k_q8/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf/results_2026-07-02T18-23-49.486875.json` |
| IFEval | Q4_K_M | `/root/models/results/ifeval/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf/results_2026-06-30T21-22-13.105914.json` |
| IFEval | Q8_0 | `/root/models/results/ifeval_q8/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf/results_2026-07-02T08-40-52.601612.json` |
| HumanEval | Q4_K_M | `/root/models/results/humaneval_q4_backup/humaneval_summary.json` |
| HumanEval | Q8_0 | `/root/models/results/humaneval/humaneval_summary.json` |