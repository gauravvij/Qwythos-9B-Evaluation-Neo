# I Benchmarked Qwythos-9B on GSM8K, IFEval, and HumanEval. Here is What I Learned.

I have been curious about Qwythos-9B since the GGUF files showed up on HuggingFace. Its a fine-tune of Qwen 3.5 9B mixed with Claude distillation data. Two quantizations were available: Q4_K_M (5.2 GB) and Q8_0 (8.9 GB). I had a 16 GB RTX 5060 Ti and a vague idea of how to set up an evaluation pipeline.

I gave [Neo](https://heyneo.com) a single prompt to handle the whole thing. No step-by-step instructions about what to install, how to configure llama.cpp, or how to format API calls. Just: run GSM8K, IFEval, and HumanEval at both quantizations and tell me what the numbers mean.

Here is what actually happened.

## The Setup

[Neo](https://heyneo.com) cloned llama.cpp, built it from source with CUDA support, and figured out the RTX 5060 Ti is a Blackwell card (compute capability 12.0) requiring specific cmake flags. Then it installed the lm_eval harness, downloaded both GGUF files, and started the llama-server.

One crucial detail: Qwen 3.5 based models split reasoning content into a separate API field. Without the `--reasoning-preserve` flag, benchmarks see blank responses. That would silently destroy every score. [Neo](https://heyneo.com) caught this during setup.

## The Eval Results

All benchmarks were run at temperature 0.0 (greedy decoding) for both quantizations. No cherry-picking, no creative reporting.

### GSM8K (Math Reasoning, 1319 samples)

- **Q4_K_M**: 80.89% (flexible-extract)
- **Q8_0**: 84.31% (flexible-extract)

A gap of only 3.4 points. The Q4 quantization preserves almost all of the model's math reasoning capability while being 70% smaller. For most applications, this is the sweet spot.

### IFEval (Instruction Following, 50 samples)

- **Q4_K_M**: 60.00% (prompt-level strict)
- **Q8_0**: 66.00% (prompt-level strict)

The largest measurable gap between quantizations at 6 points. Instruction-level metrics showed an even wider difference (+9.2pp). If you care about following formatting constraints precisely, Q8 gives you a real edge here.

There was a catch: IFEval has implicit dependencies (`langdetect` and `immutabledict`) that are not documented anywhere. They surface as ModuleNotFoundError at runtime. [Neo](https://heyneo.com) caught and fixed this during execution.

### HumanEval (Code Generation, 164 problems)

- **Both quantizations**: 0% pass@1

This is not a pipeline issue. The model simply cannot generate executable code in the format HumanEval expects. It is a reasoning and roleplay fine-tune, not a code model. The Q8 variant extracted code blocks more often (26.8% vs 21.9%), but none of those blocks passed the test cases.

I do not consider this a failure of the evaluation. It is useful data. If you need code generation, use a different model.

### The Blocked Benchmarks

HellaSwag and ARC could not be run. These are loglikelihood tasks, which require token-level probability access.

[Neo](https://heyneo.com) tried three approaches:
1. The `local-chat-completions` backend does not support loglikelihood.
2. The `local-completions` backend expects an older logprobs format that llama.cpp no longer returns.
3. The `hf` backend with a GGUF file fails because the `qwen35` architecture identifier is not yet recognized by HuggingFace Transformers' GGUF loader.

Three documented failures, zero workarounds found. This is a genuine infrastructure gap for newer model architectures.

## What Quantization Actually Means Here

The Q8 GGUF is 8.9 GB. The Q4 is 5.2 GB. That is 70% more disk space, VRAM, and bandwidth.

The measurable improvements from Q8 over Q4 are:
- GSM8K: +3.4 points
- IFEval: +6 points
- HumanEval extraction rate: +5 points

For math and instruction following, Q4 at temperature 0.0 is the practical recommendation. The Q8 premium only matters if you have headroom and need every fraction of a point on structured output compliance.

## Why This Matters for Evaluation Workflows

Most open source evaluations you see online are either cherry-picked or run with subtly wrong settings. The difference between temperature 0.6 and 0.0 on GSM8K can swing results by 10-20 points. A forgotten `--reasoning-preserve` flag makes your benchmark see empty responses. The logprobs format between llama.cpp and lm_eval is mismatched for newer models.

These are not exotic edge cases. They are routine issues that any evaluation pipeline will hit.

## How [Neo](https://heyneo.com) Handled It

I gave [Neo](https://heyneo.com) a single high-level prompt. It researched the model architecture, set up the environment, wrote custom scripts where the built-in harness fell short (HumanEval), debugged runtime failures (IFEval dependencies), caught the `--reasoning-preserve` requirement, documented the loglikelihood blockade across three backends, and produced a complete report with charts.

The full repo is at [github.com/gauravvij/Qwythos-9B-Evaluation-Neo](https://github.com/gauravvij/Qwythos-9B-Evaluation-Neo) if you want to see the raw results, the custom scripts, and the failed approaches.

## Want to Try This Yourself?

If you want to offload this kind of experimental work without hitting Claude's limits during long runs, [Neo](https://heyneo.com) is available as an MCP server for Claude Code. It lets you hand off evaluation pipelines, training runs, and debugging iterations that would otherwise eat through your context window and timeout.

Install it with `pip install neo-mcp`. Full details at [https://heyneo.com/claude-code](https://heyneo.com/claude-code).