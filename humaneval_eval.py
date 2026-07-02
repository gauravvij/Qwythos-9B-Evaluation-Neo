#!/usr/bin/env python3
"""
Fair HumanEval evaluation for reasoning models (Qwythos-9B).
Strips thinking blocks, extracts final code, uses standard pass@1.

Based on lm_eval's humaneval_instruct approach but with proper
thinking-block handling for reasoning models like Qwen3.5.
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset
from evaluate import load as load_metric


# === CONFIG ===
API_URL = "http://localhost:8081/v1/chat/completions"
MODEL_NAME = "/root/models/qwythos-9b/Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf"
OUTPUT_PATH = Path("/root/models/results/humaneval")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

GEN_KWARGS = {
    "temperature": 0.0,       # Standard for HumanEval pass@1
    "top_p": 0.95,
    "max_tokens": 2048,       # Generous to avoid truncation
    "stop": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
}


# === HELPERS ===

def strip_thinking(text: str) -> str:
    """
    Strip  blocks from Qwen3.5 reasoning models.
    These wrap the model's chain-of-thought before the answer.
    """
    # Remove  blocks
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return text.strip()


def extract_code(text: str) -> str:
    """
    Extract Python code from the model's response.
    Priority: last ```python block > last ``` block > fallback to whole text.
    For reasoning models, strips thinking tokens first.
    """
    # Strip thinking blocks first
    text = strip_thinking(text)

    # Try last ```python block
    python_blocks = list(re.finditer(r'```python\n(.*?)```', text, re.DOTALL))
    if python_blocks:
        return python_blocks[-1].group(1).strip()

    # Try any last ``` block
    all_blocks = list(re.finditer(r'```\n?(.*?)```', text, re.DOTALL))
    if all_blocks:
        return all_blocks[-1].group(1).strip()

    # No code block — return the raw text (e.g., model wrote raw code)
    return text.strip()


def build_messages(problem: dict) -> list:
    """
    Build chat messages for a HumanEval problem.
    Uses the same approach as lm_eval's humaneval_instruct:
    shows the function signature and asks for the body.
    """
    prompt = problem["prompt"]       # signature + docstring
    entry_point = problem["entry_point"]

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert Python programmer. "
            "Complete the function by writing the function body only. "
            "Use proper indentation. Return the code inside a ```python block."
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Write a solution to the following problem and "
            f"make sure that it passes the tests:\n\n"
            f"```python\n{prompt}\n```"
        )
    }

    return [system_msg, user_msg]


def call_model(messages: list) -> str:
    """Send a chat request to llama-server, return response content."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": GEN_KWARGS["temperature"],
        "top_p": GEN_KWARGS["top_p"],
        "max_tokens": GEN_KWARGS["max_tokens"],
        "stop": GEN_KWARGS["stop"],
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            return data["choices"][0]["message"]["content"], usage
        except Exception as e:
            if attempt < 2:
                print(f"  [retry {attempt+1}/3] {e}", flush=True)
                time.sleep(2)
            else:
                raise


# === MAIN ===

def main():
    print("=" * 70, flush=True)
    print("  HumanEval Evaluation — Qwythos-9B Q4_K_M (llama-server)", flush=True)
    print("  Method: instruct-style, thinking-stripped, pass@1", flush=True)
    print("=" * 70, flush=True)

    # 1. Load dataset
    print("\n[1] Loading HumanEval...", flush=True)
    dataset = load_dataset("openai/openai_humaneval", split="test")
    problems = list(dataset)
    print(f"    {len(problems)} problems loaded", flush=True)

    # 2. Verify server
    print("\n[2] Verifying server...", flush=True)
    try:
        resp = requests.get("http://localhost:8081/v1/models", timeout=5)
        resp.raise_for_status()
        print(f"    Server OK — model: {resp.json()['data'][0]['id']}", flush=True)
    except Exception as e:
        print(f"    ERROR: server unreachable: {e}", flush=True)
        sys.exit(1)

    # 3. Run inference
    print("\n[3] Running inference on 164 problems...", flush=True)
    print(f"    Gen: temp={GEN_KWARGS['temperature']}, "
          f"max_tokens={GEN_KWARGS['max_tokens']}", flush=True)
    print(f"    Stop: {GEN_KWARGS['stop']}\n", flush=True)

    results = []
    errors = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start_time = time.time()

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        messages = build_messages(problem)

        print(f"  [{i+1:3d}/164] {task_id:15s} ... ", end="", flush=True)
        t0 = time.time()

        try:
            raw_response, usage = call_model(messages)
            extracted = extract_code(raw_response)
            elapsed = time.time() - t0

            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok

            has_code = bool(extracted)
            resp_len = len(raw_response)
            ext_len = len(extracted)
            print(f"{elapsed:5.1f}s  resp={resp_len:4d}  code={ext_len:3d}  "
                  f"tok={p_tok}+{c_tok}", flush=True)

            results.append({
                "task_id": task_id,
                "prompt": problem["prompt"],
                "entry_point": problem["entry_point"],
                "test": problem["test"],
                "canonical_solution": problem["canonical_solution"],
                "raw_response": raw_response,
                "extracted_code": extracted,
                "has_code": has_code,
                "latency_seconds": elapsed,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR after {elapsed:.1f}s: {e}", flush=True)
            errors.append({"task_id": task_id, "error": str(e)})
            results.append({
                "task_id": task_id,
                "prompt": problem["prompt"],
                "entry_point": problem["entry_point"],
                "test": problem["test"],
                "canonical_solution": problem["canonical_solution"],
                "raw_response": "",
                "extracted_code": "",
                "has_code": False,
                "latency_seconds": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            })

    total_time = time.time() - start_time
    avg_time = total_time / len(results)
    extracted_count = sum(1 for r in results if r["has_code"])
    print(f"\n    Done: {len(results)} problems in "
          f"{total_time:.0f}s ({avg_time:.1f}s avg)", flush=True)
    print(f"    Code extracted: {extracted_count}/{len(results)} "
          f"({100*extracted_count/len(results):.1f}%)", flush=True)
    print(f"    Total tokens: {total_prompt_tokens} prompt + "
          f"{total_completion_tokens} completion", flush=True)

    # 4. Evaluate pass@1
    print("\n[4] Computing pass@1...", flush=True)
    predictions = []
    references = []

    for r in results:
        if r["extracted_code"]:
            # Standard HumanEval: prompt (signature) + body
            full_code = r["prompt"] + "\n" + r["extracted_code"]
            predictions.append([full_code])
        else:
            predictions.append([r["prompt"] + "\n    pass"])

        # Reference: test code that calls check(entry_point)
        references.append(r["test"] + f"\ncheck({r['entry_point']})")

    # Verify predictions look sane with a sample
    print(f"    Sample prediction (problem 0):")
    print(f"      {predictions[0][0][:150]}...", flush=True)
    print(f"    Sample reference (problem 0):")
    print(f"      {references[0][:150]}...", flush=True)

    code_eval = load_metric("code_eval")
    eval_results = code_eval.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )
    pass_at_1 = eval_results[0]["pass@1"]
    print(f"\n    >>> HumanEval pass@1: {pass_at_1:.4f} "
          f"({pass_at_1*100:.2f}%)", flush=True)

    # 5. Also show "extraction rate" as a diagnostic
    # If the model always outputs thinking but no code, extraction rate is 0
    print(f"\n    >>> Code extraction rate: {extracted_count}/{len(results)} "
          f"({100*extracted_count/len(results):.1f}%)", flush=True)
    if pass_at_1 == 0 and extracted_count > 0:
        print("    (Model generates code but tests fail — genuine accuracy issue)")
    elif pass_at_1 == 0 and extracted_count == 0:
        print("    (Model never wrote valid code — possible template/format issue)")

    # 6. Save results
    print("\n[5] Saving...", flush=True)

    # Save minimal JSON (without raw responses for size)
    summary = {
        "model": MODEL_NAME,
        "api_url": API_URL,
        "gen_kwargs": GEN_KWARGS,
        "method": "instruct + thinking-stripped + pass@1",
        "num_problems": len(results),
        "num_errors": len(errors),
        "pass_at_1": pass_at_1,
        "extraction_rate": extracted_count / len(results),
        "total_time_seconds": total_time,
        "avg_time_per_problem_seconds": avg_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }

    with open(OUTPUT_PATH / "humaneval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Summary: {OUTPUT_PATH / 'humaneval_summary.json'}")

    # Save detailed results separately
    detailed = {k: v for k, v in summary.items() if k != "api_url"}
    detailed["results"] = [
        {
            "task_id": r["task_id"],
            "extracted_code": r["extracted_code"],
            "has_code": r["has_code"],
            "raw_response": r["raw_response"][:500] + ("..." if len(r["raw_response"]) > 500 else ""),
            "latency_seconds": r["latency_seconds"],
        }
        for r in results
    ]
    detailed["errors"] = errors

    with open(OUTPUT_PATH / "humaneval_detailed.json", "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"    Detailed: {OUTPUT_PATH / 'humaneval_detailed.json'}")

    # 7. Print report
    print("\n" + "=" * 70, flush=True)
    print("  FINAL REPORT", flush=True)
    print("=" * 70, flush=True)
    print(f"  Model:          Qwythos-9B Q4_K_M (llama-server on :8081)", flush=True)
    print(f"  Method:         instruct-style + thinking-stripped + pass@1", flush=True)
    print(f"  Temperature:    0.0 (greedy, standard for pass@1)", flush=True)
    print(f"  Max tokens:     2048", flush=True)
    print(f"  Problems:       {len(results)}", flush=True)
    print(f"  Errors:         {len(errors)}", flush=True)
    print(f"  Code extracted: {extracted_count}/{len(results)} "
          f"({100*extracted_count/len(results):.1f}%)", flush=True)
    print(f"  ─────────────────────────────────────────────", flush=True)
    print(f"  >> pass@1:      {pass_at_1:.4f} ({pass_at_1*100:.2f}%)", flush=True)
    print(f"  ─────────────────────────────────────────────", flush=True)
    print(f"  Total time:     {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"  Avg per problem: {avg_time:.1f}s", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()