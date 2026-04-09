import argparse
from pathlib import Path

import torch

from core import run_benchmark

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Self-contained benchmark for newer Qwen/Llama model pairs.")
    parser.add_argument("--pairs_file", type=Path, default=SCRIPT_DIR / "candidate_pairs.json")
    parser.add_argument("--output_root", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--pair_names", nargs="*", default=None)
    parser.add_argument("--pair_tiers", nargs="*", default=["default"])
    parser.add_argument("--prompt_sets", nargs="*", default=["core"])
    parser.add_argument("--prompts_per_set", type=int, default=2)
    parser.add_argument("--prompt_file", type=Path, default=None)
    parser.add_argument("--gammas", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--top_ks", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--top_ps", nargs="+", type=float, default=[0.0, 0.9])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--approx_quant", type=str, default="none", choices=["none", "int8", "nf4", "fp4"])
    parser.add_argument("--target_quant", type=str, default="none", choices=["none", "int8", "nf4", "fp4"])
    parser.add_argument("--compile_draft", action="store_true", default=False)
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--run_small_baseline", action="store_true", default=False)
    parser.add_argument("--fail_on_incompatible_pairs", action="store_true", default=False)
    parser.add_argument("--fail_on_pair_error", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
