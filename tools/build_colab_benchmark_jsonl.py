import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

CNN_DM_TEST_PARQUET = "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/test-00000-of-00001.parquet"
HUMANEVAL_TEST_PARQUET = "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet"
GSM8K_TEST_PARQUET = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"


def sample_indices(length, count, seed):
    rng = random.Random(seed)
    indices = list(range(length))
    rng.shuffle(indices)
    return sorted(indices[:count])


def trim_text(text, max_chars):
    text = text.strip()
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def make_cnn_dm_prompt(example, max_chars):
    article = trim_text(example["article"], max_chars)
    return f"Summarize the following article in 3 concise bullet points:\n\n{article}"


def make_humaneval_prompt(example):
    prompt = example["prompt"].rstrip()
    return f"Complete the following Python function:\n\n```python\n{prompt}\n```"


def make_gsm8k_prompt(example):
    question = example["question"].strip()
    return f"Solve the following math problem step by step:\n\n{question}"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a compact Colab benchmark JSONL for search-v2.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cnn_count", type=int, default=10)
    parser.add_argument("--code_count", type=int, default=10)
    parser.add_argument("--math_count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cnn_max_chars", type=int, default=4000)
    args = parser.parse_args()

    rows = []

    cnn_dm = load_dataset("parquet", data_files={"test": CNN_DM_TEST_PARQUET}, split="test")
    for idx in sample_indices(len(cnn_dm), args.cnn_count, args.seed):
        example = cnn_dm[idx]
        rows.append(
            {
                "task": "cnn_dailymail",
                "source": "ccdv/cnn_dailymail",
                "id": f"cnn_dailymail_{idx}",
                "prompt": make_cnn_dm_prompt(example, args.cnn_max_chars),
            }
        )

    humaneval = load_dataset("parquet", data_files={"test": HUMANEVAL_TEST_PARQUET}, split="test")
    for idx in sample_indices(len(humaneval), args.code_count, args.seed + 1):
        example = humaneval[idx]
        rows.append(
            {
                "task": "humaneval",
                "source": "openai/openai_humaneval",
                "id": f"humaneval_{idx}",
                "prompt": make_humaneval_prompt(example),
            }
        )

    gsm8k = load_dataset("parquet", data_files={"test": GSM8K_TEST_PARQUET}, split="test")
    for idx in sample_indices(len(gsm8k), args.math_count, args.seed + 2):
        example = gsm8k[idx]
        rows.append(
            {
                "task": "gsm8k",
                "source": "openai/gsm8k",
                "id": f"gsm8k_{idx}",
                "prompt": make_gsm8k_prompt(example),
            }
        )

    write_jsonl(args.output, rows)
    print(json.dumps({
        "output": str(args.output),
        "total_rows": len(rows),
        "cnn_count": sum(1 for r in rows if r["task"] == "cnn_dailymail"),
        "code_count": sum(1 for r in rows if r["task"] == "humaneval"),
        "math_count": sum(1 for r in rows if r["task"] == "gsm8k"),
    }, indent=2))


if __name__ == "__main__":
    main()
