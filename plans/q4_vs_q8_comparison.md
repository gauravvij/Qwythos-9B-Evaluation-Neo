# Q4_K_M vs Q8_0 Comparison: Full Benchmark Re-Run

## Goal
Re-run GSM8K, IFEval, and HumanEval with Q8_0 quantization, then produce a detailed Q4 vs Q8 comparison report.

## Approach
1. Download the Q8_0 GGUF file (empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF, Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf)
2. Stop current llama-server (Q4_K_M), restart with Q8_0
3. Run GSM8K full (1319 samples) via lm_eval local-chat-completions
4. Run IFEval (limit=50) via lm_eval local-chat-completions
5. Run HumanEval (164 problems) via custom script
6. Compile comparison report with side-by-side metrics

## Subtasks

1. **Download Q8_0 GGUF** — Use huggingface-hub (`hf download`) or direct curl to download `Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf` to `/root/models/qwythos-9b/`
   - Expected output: `/root/models/qwythos-9b/Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf` (~9.4 GiB)
   - verify: file exists, size is ~9-10 GiB, has readable GGUF header

2. **Kill Q4 llama-server, start Q8 llama-server** — Kill existing llama-server on port 8081 (it's serving Q4_K_M). Restart with the Q8_0 GGUF: same args (`-ngl 99 --port 8081 --host 0.0.0.0 --reasoning-preserve`)
   - verify: `curl http://localhost:8081/v1/models` returns the new Q8_0 model ID

3. **Run GSM8K full (1319 problems)** — Using lm_eval:
   ```bash
   source /root/models/.venv/bin/activate && \
   lm_eval --model local-chat-completions \
     --model_args 'model=/root/models/qwythos-9b/Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf,base_url=http://localhost:8081/v1/chat/completions' \
     --tasks gsm8k \
     --apply_chat_template \
     --gen_kwargs '{"temperature":0.6,"top_p":0.95,"top_k":20,"repeat_penalty":1.05}' \
     --output_path /root/models/results/gsm8k_q8 \
     --num_fewshot 5
   ```
   - Expected output: results JSON in `/root/models/results/gsm8k_q8/`
   - verify: script exits 0, results file has non-null `results.gsm8k.exact_match,flexible-extract` and `results.gsm8k.exact_match,strict-match`

4. **Run IFEval (limit=50)** — Using lm_eval:
   ```bash
   source /root/models/.venv/bin/activate && \
   lm_eval --model local-chat-completions \
     --model_args 'model=/root/models/qwythos-9b/Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf,base_url=http://localhost:8081/v1/chat/completions' \
     --tasks ifeval \
     --apply_chat_template \
     --gen_kwargs '{"temperature":0.6,"top_p":0.95,"top_k":20,"repeat_penalty":1.05}' \
     --output_path /root/models/results/ifeval_q8 \
     --limit 50
   ```
   - Expected output: results JSON in `/root/models/results/ifeval_q8/`
   - verify: script exits 0, results file has non-null ifeval metrics

5. **Run HumanEval (164 problems)** — Custom script:
   ```bash
   source /root/models/.venv/bin/activate && \
   python3 /root/models/humaneval_eval.py
   ```
   (The script already exists at `/root/models/humaneval_eval.py` and reads `model` from its own config. Need to update the MODEL_NAME variable to point to the Q8_0 file before running.)
   - Expected output: `/root/models/results/humaneval/` — will overwrite Q4 results
   - verify: summary file has non-null pass_at_1

6. **Compile comparison report** — Read all Q4 and Q8 results files, produce `/root/models/reports/qwythos_9b_q4_vs_q8_comparison.md` with side-by-side table, deltas, and analysis

## Deliverables
| File | Description |
|------|-------------|
| `/root/models/qwythos-9b/Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf` | Q8_0 GGUF file |
| `/root/models/results/gsm8k_q8/...` | GSM8K Q8 results |
| `/root/models/results/ifeval_q8/...` | IFEval Q8 results |
| `/root/models/results/humaneval/humaneval_summary.json` | HumanEval Q8 results (overwrites Q4) |
| `/root/models/reports/qwythos_9b_q4_vs_q8_comparison.md` | Side-by-side comparison report |

## Evaluation Criteria
- All 3 benchmarks complete successfully (GSM8K, IFEval, HumanEval)
- Comparison report produced with Q4 and Q8 scores for each benchmark
- Delta (Q8 - Q4) reported for every metric