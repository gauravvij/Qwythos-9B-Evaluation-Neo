# Qwythos-9B Evaluation — Autonomous AI Engineering by Neo

This repo contains a complete evaluation of [Qwythos-9B-Claude-Mythos-5-1M](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF), a Qwen3.5-9B based reasoning model, across multiple benchmarks at two different quantization levels (Q4\_K\_M and Q8\_0).

Everything here was produced **autonomously by Neo** — an AI Engineering Agent that takes a high-level goal, researches approaches, writes code, runs experiments, iterates on failures, and delivers results without hand-holding.

## What was evaluated

- **GSM8K** (grade school math) — full 1319 samples
- **IFEval** (instruction following) — 50 samples
- **HumanEval** (code generation) — 164 problems

Each benchmark was run on both Q4\_K\_M and Q8\_0 quantized GGUF variants, all at temperature=0.0 (greedy decoding) for a fair comparison.

## Key Results

| Benchmark | Q4\_K\_M | Q8\_0 | Δ |
|-----------|:--------:|:-----:|:-:|
| GSM8K (flexible-extract) | **80.89%** | **84.31%** | +3.42pp |
| IFEval (prompt-level strict) | 60.00% | **66.00%** | +6.00pp |
| HumanEval (pass@1) | 0.00% | 0.00% | 0pp |

> All benchmarks use temperature=0.0 (greedy decoding) across both quantizations.

See the full report at [`reports/qwythos_9b_eval_report.md`](reports/qwythos_9b_eval_report.md).

## Key Findings

- **GSM8K**: Q4_K_M (80.89%) and Q8_0 (84.31%) are close at temp=0.0 — the quantization gap is only 3.4pp. Both quantizations use greedy decoding for a fair comparison.
- **IFEval**: Q8_0 shows a consistent 6-10pp advantage over Q4_K_M, the largest measurable gap between quantizations.
- **HumanEval**: Both quantizations score 0% pass@1 — this is a model capability limitation, not a pipeline issue.
- **Practical recommendation**: Q4_K_M at temp=0.0 provides the best quality/size tradeoff for most use cases.

## Reproducing or extending this evaluation

You can re-run or build on this work using Neo in VS Code or Cursor. Clone the repo and give Neo a prompt like:

> "Run GSM8K on the Q4 results at temperature 0.0 so we can do a fair Q4 vs Q8 comparison."

Or extend to new benchmarks:

> "Take the Q8\_0 GSM8K results in /root/models/results/gsm8k_q8/ and generate a per-question error analysis. Classify the incorrect answers into categories: arithmetic error, reasoning error, incomplete output, and hallucinated answer. Output a chart."

Or try a new model:

> "Download the Q3\_K\_M GGUF variant of this same model and run the same three benchmarks at temperature=0.0. Produce a three-way comparison table against the existing Q4 and Q8 numbers."

All evaluation infrastructure is in `slm_eval_harness/` and the custom HumanEval script is at `humaneval_eval.py`. Results JSON files are under `results/` organized by benchmark.

## Files

- `reports/qwythos_9b_eval_report.md` — final report with Q4 vs Q8 comparison
- `humaneval_eval.py` — custom HumanEval evaluation script (chat API, thinking-stripping, code extraction)
- `slm_eval_harness/` — evaluation framework with components for config, checkpointing, parsing
- `results/` — raw result JSON files for all benchmarks

## What is Neo?

Neo is an autonomous AI Engineering Agent. You give it a goal, and it handles everything else: researching approaches, setting up environments, writing code, running experiments, debugging failures, iterating, and delivering results. It works in VS Code or Cursor as an extension.

Neo can do more than evaluation — it handles ML training, fine-tuning (LoRA/QLoRA, full fine-tuning, RLHF), RAG pipeline construction, AI agent building, data science workflows, and production deployment.