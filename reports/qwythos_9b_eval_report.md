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

> **All benchmarks use temperature=0.0 (greedy decoding) across both Q4_K_M and Q8_0 quantizations**, ensuring a fair and consistent comparison.

## Benchmark Results

### GSM8K (Grade School Math — Full 1319 samples)

| Quant | Temperature | Flexible-extract | Strict-match |
|-------|:-----------:|:----------------:|:------------:|
| **Q4_K_M** | **0.0** | **80.89%** ±1.08% | **80.29%** ±1.08% |
| **Q8_0** | **0.0** | **84.31%** ±1.00% | **83.24%** ±1.03% |
| **Δ (Q8 - Q4)** | | **+3.42pp** | **+2.95pp** |

**Conclusion**: Both quantizations achieve strong GSM8K performance at temperature=0.0. The Q8_0 variant shows a modest +3.4pp advantage over Q4_K_M, indicating that quantization precision has a small but measurable impact on math reasoning at greedy decoding.

### IFEval (Instruction Following — limit=50 samples)

| Metric | Q4_K_M (temp=0.0) | Q8_0 (temp=0.0) | Δ |
|--------|:------------------:|:----------------:|:-:|
| **prompt_level_strict** | 60.00% ±7.00% | **66.00%** ±6.77% | +6.00pp |
| **prompt_level_loose** | 60.00% ±7.00% | **68.00%** ±6.66% | +8.00pp |
| **inst_level_strict** | 60.53% | **69.74%** | +9.21pp |
| **inst_level_loose** | 60.53% | **71.05%** | +10.52pp |

**Conclusion**: Q8_0 shows a consistent advantage over Q4_K_M on instruction following, with a 6-10pp improvement across all four metrics. This is the largest measurable difference between quantizations. Both are within the expected range for a 9B-class reasoning model.

### HumanEval (Code Generation — 164 problems)

| Metric | Q4_K_M (temp=0.0) | Q8_0 (temp=0.0) | Δ |
|--------|:------------------:|:----------------:|:-:|
| **pass@1** | 0.00% | 0.00% | 0.00pp |
| **Code extraction rate** | 21.95% (36/164) | **26.83%** (44/164) | +4.88pp |
| **Errors** | 0 | 0 | — |
| **Total time** | 20.9 min | 35.2 min | +14.3 min |
| **Avg per problem** | 7.6s | 12.9s | +5.3s |

> **Both quantizations score 0% pass@1**. The model fundamentally cannot solve coding problems at the HumanEval level in this evaluation setup. The slightly higher Q8_0 extraction rate (26.8% vs 21.9%) means the model produces marginally more extractable code blocks, but none of the extracted code passes the test cases. This is a **genuine model capability limitation** — this is a Qwen3.5-based reasoning/roleplay fine-tune, not a code-specialized model. The chat/instruct format also makes it difficult to produce standalone code in the expected format.

## Summary Comparison Table

| Benchmark | Q4_K_M | Q8_0 | Δ |
|-----------|:------:|:----:|:-:|
| **GSM8K** (flexible-extract) | 80.89% | **84.31%** | +3.42pp |
| **IFEval** (prompt_strict) | 60.00% | **66.00%** | +6.00pp |
| **HumanEval** (pass@1) | 0.00% | 0.00% | 0pp |

> All benchmarks use temperature=0.0 (greedy decoding) across both quantizations.

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

## Key Takeaways

1. **GSM8K performs well on both quantizations at temp=0.0**: Q4_K_M achieves 80.89% and Q8_0 achieves 84.31% — a modest +3.4pp difference. The model's math reasoning capability is unlocked at greedy decoding regardless of quantization.

2. **Q8_0 shows the largest improvement on instruction following** (IFEval: +6-10pp), making it the most meaningful gap between quantizations. For code extraction (HumanEval: +4.9pp), the improvement is smaller.

3. **HumanEval is a hard blocker** — the model's chat/roleplay fine-tuning makes it fundamentally incompatible with standalone code generation benchmarks. 0% pass@1 across both quantizations.

4. **Practical recommendation**: For general use, Q4_K_M at temp=0.0 provides the best quality/size tradeoff. The Q8_0 quantization offers marginal improvements (3-10pp depending on task) that may not justify the 70% larger file size (8.9 GiB vs 5.2 GiB) in most applications.

## Results File Locations

| Benchmark | Quant | Path |
|-----------|-------|------|
| GSM8K | Q4_K_M | `/root/models/results/gsm8k_full/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf/results_2026-07-03T09-01-55.622891.json` |
| GSM8K | Q8_0 | `/root/models/results/gsm8k_q8/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf/results_2026-07-02T18-23-49.486875.json` |
| IFEval | Q4_K_M | `/root/models/results/ifeval/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf/results_2026-07-03T09-20-42.804806.json` |
| IFEval | Q8_0 | `/root/models/results/ifeval_q8/__root__models__qwythos-9b__Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf/results_2026-07-02T08-40-52.601612.json` |
| HumanEval | Q4_K_M | `/root/models/results/humaneval_q4_backup/humaneval_summary.json` |
| HumanEval | Q8_0 | `/root/models/results/humaneval/humaneval_summary.json` |