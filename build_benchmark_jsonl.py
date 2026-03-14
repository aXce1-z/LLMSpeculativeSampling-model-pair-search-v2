import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def make_summarization_prompt(example, source_name):
    if source_name == "cnn_dailymail":
        article = example["article"].strip()
        return f"Summarize the following article in 3 concise bullet points:\n\n{article}"
    if source_name == "xsum":
        document = example["document"].strip()
        return f"Write a one-paragraph summary of the following article:\n\n{document}"
    raise ValueError(source_name)


def make_humaneval_prompt(example):
    prompt = example["prompt"].rstrip()
    return f"Complete the following Python function:\n\n```python\n{prompt}\n```"


def make_gsm8k_prompt(example):
    question = example["question"].strip()
    return f"Solve the following math problem step by step:\n\n{question}"


def sample_indices(length, count, seed):
    rng = random.Random(seed)
    idxs = list(range(length))
    rng.shuffle(idxs)
    return sorted(idxs[:count])


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a fixed JSONL benchmark prompt set from public datasets.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summ_source", choices=["cnn_dailymail", "xsum"], default="cnn_dailymail")
    parser.add_argument("--summ_count", type=int, default=20)
    parser.add_argument("--code_count", type=int, default=20)
    parser.add_argument("--math_count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = []

    if args.summ_source == "cnn_dailymail":
        ds = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    else:
        ds = load_dataset("GEM/xsum", split="test", trust_remote_code=True)
    for i in sample_indices(len(ds), args.summ_count, args.seed):
        ex = ds[i]
        rows.append({
            "task": "summarization",
            "source": args.summ_source,
            "id": f"{args.summ_source}_{i}",
            "prompt": make_summarization_prompt(ex, args.summ_source),
        })

    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    for i in sample_indices(len(ds), args.code_count, args.seed + 1):
        ex = ds[i]
        rows.append({
            "task": "code",
            "source": "openai_humaneval",
            "id": f"humaneval_{i}",
            "prompt": make_humaneval_prompt(ex),
        })

    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    for i in sample_indices(len(ds), args.math_count, args.seed + 2):
        ex = ds[i]
        rows.append({
            "task": "math",
            "source": "gsm8k",
            "id": f"gsm8k_{i}",
            "prompt": make_gsm8k_prompt(ex),
        })

    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()

