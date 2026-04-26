
import csv
import itertools
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

from prompt_sets import get_prompt_sets


def resolve_dtype(dtype_name):
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def model_device(model):
    return next(model.parameters()).device


def sync_if_needed(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_segment(device_name, fn, *args, **kwargs):
    sync_if_needed(device_name)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    sync_if_needed(device_name)
    return result, time.perf_counter() - t0


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter_vals[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter_mask = cumulative_probs > top_p
        filter_mask[..., 1:] = filter_mask[..., :-1].clone()
        filter_mask[..., 0] = 0
        indices_to_remove = filter_mask.scatter(1, sorted_indices, filter_mask)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=1)


def sample(probs: torch.Tensor, num_samples: int = 1):
    return torch.multinomial(probs, num_samples=num_samples)


def max_fn(x: torch.Tensor):
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    fallback = torch.full_like(x_max, 1.0 / x_max.shape[1])
    return torch.where(x_max_sum > 0, x_max / x_max_sum, fallback)


def align_prob_row(prob_row: torch.Tensor, target_vocab_size: int) -> torch.Tensor:
    current_vocab_size = prob_row.shape[1]
    if current_vocab_size == target_vocab_size:
        return prob_row
    if current_vocab_size > target_vocab_size:
        return prob_row[:, :target_vocab_size]
    pad = torch.zeros(
        (prob_row.shape[0], target_vocab_size - current_vocab_size),
        dtype=prob_row.dtype,
        device=prob_row.device,
    )
    return torch.cat((prob_row, pad), dim=1)


def split_kv(kv):
    if isinstance(kv, (tuple, list)) and len(kv) >= 2:
        return kv[0], kv[1]
    raise ValueError(f"Unexpected kv cache entry type: {type(kv)}")


@dataclass
class KVCacheStats:
    generate_calls: int = 0
    forward_calls: int = 0
    prefill_forward_calls: int = 0
    decode_forward_calls: int = 0
    initial_prompt_prefill_forward_calls: int = 0
    sampled_tokens: int = 0
    prefill_tokens_total: int = 0
    decode_tokens_total: int = 0
    initial_prompt_prefill_tokens: int = 0
    total_generate_ms: float = 0.0
    total_forward_ms: float = 0.0
    total_prefill_forward_ms: float = 0.0
    total_decode_forward_ms: float = 0.0
    initial_prompt_prefill_forward_ms: float = 0.0
    total_sample_ms: float = 0.0
    total_concat_ms: float = 0.0
    total_rollback_ms: float = 0.0


def _preserve_cache_object(past_key_values):
    return past_key_values


class KVCacheModel:
    def __init__(self, model, temperature=1.0, top_k=0, top_p=0.0):
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.stats = KVCacheStats()

    def _normalize_logits_tensor(self, logits: torch.Tensor) -> torch.Tensor:
        flat = logits.reshape(-1, logits.shape[-1])
        flat = norm_logits(flat, self._temperature, self._top_k, self._top_p)
        return flat.reshape_as(logits)

    def _forward_with_kvcache(self, input_ids: torch.Tensor):
        if self._past_key_values is None:
            t0 = time.perf_counter()
            outputs = self._model(input_ids, use_cache=True)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.stats.forward_calls += 1
            self.stats.prefill_forward_calls += 1
            self.stats.prefill_tokens_total += int(input_ids.shape[1])
            self.stats.total_forward_ms += dt_ms
            self.stats.total_prefill_forward_ms += dt_ms
            self.stats.initial_prompt_prefill_forward_calls += 1
            self.stats.initial_prompt_prefill_tokens += int(input_ids.shape[1])
            self.stats.initial_prompt_prefill_forward_ms += dt_ms
            self._prob_history = self._normalize_logits_tensor(outputs.logits)
            self._past_key_values = _preserve_cache_object(outputs.past_key_values)
            return self._prob_history[:, -1, :]

        cached_len = self._prob_history.shape[1]
        last_input_id = input_ids[:, cached_len:]
        if last_input_id.dim() == 1:
            last_input_id = torch.unsqueeze(last_input_id, 0)
        if last_input_id.numel() == 0:
            return self._prob_history[:, -1, :]

        t0 = time.perf_counter()
        outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        uncached_tokens = int(last_input_id.shape[1])
        self.stats.forward_calls += 1
        self.stats.total_forward_ms += dt_ms
        if uncached_tokens > 1:
            self.stats.prefill_forward_calls += 1
            self.stats.prefill_tokens_total += uncached_tokens
            self.stats.total_prefill_forward_ms += dt_ms
        else:
            self.stats.decode_forward_calls += 1
            self.stats.decode_tokens_total += uncached_tokens
            self.stats.total_decode_forward_ms += dt_ms
        not_cached_q = outputs.logits
        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0)
        not_cached_q = self._normalize_logits_tensor(not_cached_q)
        self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
        self._past_key_values = _preserve_cache_object(outputs.past_key_values)
        return not_cached_q[:, -1, :]

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, gamma: int):
        t_generate0 = time.perf_counter()
        self.stats.generate_calls += 1
        x = prefix
        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            t_sample0 = time.perf_counter()
            next_tok = sample(q)
            self.stats.total_sample_ms += (time.perf_counter() - t_sample0) * 1000.0
            self.stats.sampled_tokens += 1
            t_cat0 = time.perf_counter()
            x = torch.cat((x, next_tok), dim=1)
            self.stats.total_concat_ms += (time.perf_counter() - t_cat0) * 1000.0
        self.stats.total_generate_ms += (time.perf_counter() - t_generate0) * 1000.0
        return x

    @torch.no_grad()
    def rollback(self, end_pos: int):
        t0 = time.perf_counter()
        if self._past_key_values is None:
            self.stats.total_rollback_ms += (time.perf_counter() - t0) * 1000.0
            return
        if hasattr(self._past_key_values, "crop"):
            try:
                self._past_key_values.crop(end_pos)
                self._prob_history = self._prob_history[:, :end_pos, :]
                self.stats.total_rollback_ms += (time.perf_counter() - t0) * 1000.0
                return
            except ValueError:
                # Some modern cache layers (for example sliding-window layers used by Gemma 4)
                # cannot be cropped after the active cache window has moved. Fall back to forcing
                # a cache rebuild from the preserved prefix on the next forward pass.
                self._past_key_values = None
                self._prob_history = self._prob_history[:, :end_pos, :]
                self.stats.total_rollback_ms += (time.perf_counter() - t0) * 1000.0
                return
        trimmed = []
        for kv in self._past_key_values:
            k, v = split_kv(kv)
            if isinstance(self._model, BloomForCausalLM):
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
            else:
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
            trimmed.append((k, v))
        self._past_key_values = trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]
        self.stats.total_rollback_ms += (time.perf_counter() - t0) * 1000.0

@torch.no_grad()
def autoregressive_sampling(x, model, num_tokens, temperature=1.0, top_k=0, top_p=0.0):
    n = x.shape[1]
    target_len = n + num_tokens
    past_key_values = None
    while n < target_len:
        if past_key_values is None:
            outputs = model(x, use_cache=True)
        else:
            outputs = model(x[:, -1:], past_key_values=past_key_values, use_cache=True)
        last_p = norm_logits(outputs.logits[:, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        n += 1
    return x


def speculative_sampling_with_stats(prefix, approx_model, target_model, max_len, gamma=4, temperature=1.0, top_k=0, top_p=0.0, random_seed=None):
    seq_len = prefix.shape[1]
    total_len = seq_len + max_len
    approx_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    device_name = model_device(approx_model).type
    accepted_count = 0
    resample_count = 0
    target_sample_count = 0
    rounds = 0
    accepted_per_round = []
    contributed_per_round = []
    draft_generate_ms_per_round = []
    target_verify_ms_per_round = []

    while prefix.shape[1] < total_len:
        rounds += 1
        prefix_len = prefix.shape[1]
        x, draft_elapsed = timed_segment(device_name, approx_cache.generate, prefix, gamma)
        _, target_elapsed = timed_segment(device_name, target_cache.generate, x, 1)
        draft_generate_ms_per_round.append(draft_elapsed * 1000.0)
        target_verify_ms_per_round.append(target_elapsed * 1000.0)
        n = prefix_len + gamma - 1
        accepted_this_round = 0

        for i in range(gamma):
            if random_seed is not None:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device=model_device(target_model))
            j = x[:, prefix_len + i]
            approx_row = align_prob_row(
                approx_cache._prob_history[:, prefix_len + i - 1, :],
                target_cache._prob_history.shape[-1],
            )
            accept_prob = target_cache._prob_history[:, prefix_len + i - 1, j] / approx_row[:, j]
            if r > accept_prob:
                n = prefix_len + i - 1
                break
            accepted_count += 1
            accepted_this_round += 1

        prefix = x[:, :n + 1]
        approx_cache.rollback(n + 1)
        if n < prefix_len + gamma - 1:
            approx_row = align_prob_row(
                approx_cache._prob_history[:, n, :],
                target_cache._prob_history.shape[-1],
            )
            t = sample(max_fn(target_cache._prob_history[:, n, :] - approx_row))
            resample_count += 1
            target_cache.rollback(n + 1)
        else:
            t = sample(target_cache._prob_history[:, -1, :])
            target_sample_count += 1
            target_cache.rollback(n + 2)

        prefix = torch.cat((prefix, t), dim=1)
        accepted_per_round.append(accepted_this_round)
        contributed_per_round.append(prefix.shape[1] - prefix_len)

    generated_tokens = prefix.shape[1] - seq_len
    proposed_draft_tokens = rounds * gamma
    stats = {
        "rounds": rounds,
        "accepted_count": accepted_count,
        "resample_count": resample_count,
        "target_sample_count": target_sample_count,
        "generated_tokens": generated_tokens,
        "proposed_draft_tokens": proposed_draft_tokens,
        "acceptance_rate": accepted_count / proposed_draft_tokens if proposed_draft_tokens else 0.0,
        "all_accept_round_rate": target_sample_count / rounds if rounds else 0.0,
        "resample_round_rate": resample_count / rounds if rounds else 0.0,
        "avg_accepted_per_round": statistics.mean(accepted_per_round) if accepted_per_round else 0.0,
        "avg_contributed_per_round": statistics.mean(contributed_per_round) if contributed_per_round else 0.0,
        "avg_draft_generate_ms_per_round": statistics.mean(draft_generate_ms_per_round) if draft_generate_ms_per_round else 0.0,
        "avg_target_verify_ms_per_round": statistics.mean(target_verify_ms_per_round) if target_verify_ms_per_round else 0.0,
        "avg_draft_prefill_forward_ms_per_round": approx_cache.stats.total_prefill_forward_ms / rounds if rounds else 0.0,
        "avg_draft_decode_forward_ms_per_round": approx_cache.stats.total_decode_forward_ms / rounds if rounds else 0.0,
        "avg_incremental_prefill_forward_ms_per_round": (
            max(approx_cache.stats.total_prefill_forward_ms - approx_cache.stats.initial_prompt_prefill_forward_ms, 0.0) / rounds
            if rounds else 0.0
        ),
        "draft_generate_ms_total": sum(draft_generate_ms_per_round),
        "target_verify_ms_total": sum(target_verify_ms_per_round),
        "draft_prefill_forward_ms_total": approx_cache.stats.total_prefill_forward_ms,
        "draft_decode_forward_ms_total": approx_cache.stats.total_decode_forward_ms,
        "incremental_prefill_forward_ms_total": max(
            approx_cache.stats.total_prefill_forward_ms - approx_cache.stats.initial_prompt_prefill_forward_ms, 0.0
        ),
    }
    return prefix, stats


def load_pairs(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def select_pairs(pairs, pair_names=None, pair_tiers=None):
    selected = []
    pair_names = set(pair_names or [])
    pair_tiers = set(pair_tiers or [])
    for pair in pairs:
        if pair_names and pair["name"] not in pair_names:
            continue
        if pair_tiers and pair.get("tier") not in pair_tiers and not pair_names:
            continue
        selected.append(pair)
    return selected


def load_prompts(args):
    prompts = []
    if args.prompt_file:
        for line in args.prompt_file.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            prompts.append({"task": data.get("task", "custom"), "prompt": data["prompt"]})
        return prompts

    selected = get_prompt_sets(args.prompt_sets, args.prompts_per_set)
    for task, items in selected.items():
        for idx, prompt in enumerate(items):
            prompts.append({"task": task, "prompt_id": f"{task}_{idx}", "prompt": prompt})
    return prompts

def resolve_quantization(quant_mode, compute_dtype):
    quant_mode = (quant_mode or "none").lower()
    if quant_mode == "none":
        return None
    if quant_mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant_mode in {"nf4", "fp4"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_mode,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported quantization mode: {quant_mode}")


def _should_retry_with_remote_code(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "does not recognize this architecture" in msg
        or "model type" in msg and "trust_remote_code" in msg
        or "model type `qwen3`" in msg
    )


def load_tokenizer(tokenizer_source, trust_remote_code):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        if trust_remote_code or not _should_retry_with_remote_code(exc):
            raise
        print(f"[tokenizer.warn] retrying {tokenizer_source} with trust_remote_code=True")
        return AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)


def load_model(model_name, dtype, device, trust_remote_code, quant_mode, compile_model=False, compile_mode="reduce-overhead"):
    kwargs = {"trust_remote_code": trust_remote_code}
    quant_cfg = resolve_quantization(quant_mode, dtype)
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
    else:
        kwargs["torch_dtype"] = dtype
    if device == "cuda":
        kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except ValueError as exc:
        if trust_remote_code or not _should_retry_with_remote_code(exc):
            raise
        retry_kwargs = dict(kwargs)
        retry_kwargs["trust_remote_code"] = True
        print(f"[model.warn] retrying {model_name} with trust_remote_code=True")
        model = AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    if compile_model:
        if device != "cuda":
            raise RuntimeError("draft compile requested but CUDA is unavailable in this environment.")
        if quant_cfg is not None:
            raise RuntimeError("draft compile is currently unsupported when approx quantization is enabled.")
        if not hasattr(torch, "compile"):
            raise RuntimeError("draft compile requested but torch.compile is unavailable in this environment.")
        model = torch.compile(model, mode=compile_mode)
    return model


def output_vocab_size(model):
    emb = model.get_output_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return int(emb.weight.shape[0])
    cfg_vocab = getattr(model.config, "vocab_size", None)
    if isinstance(cfg_vocab, int):
        return cfg_vocab
    return None


def effective_context_limit(tokenizer, small_model, large_model):
    limits = []
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 0 < tok_limit < 1_000_000:
        limits.append(tok_limit)
    for model in (small_model, large_model):
        cfg_limit = getattr(model.config, "max_position_embeddings", None)
        if isinstance(cfg_limit, int) and cfg_limit > 0:
            limits.append(cfg_limit)
    if not limits:
        return None
    return min(limits)


def encode_prompt(tokenizer, prompt_text, device, small_model, large_model, reserve_tokens=0):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    context_limit = effective_context_limit(tokenizer, small_model, large_model)
    if context_limit is not None:
        usable_limit = max(1, context_limit - max(0, reserve_tokens))
        if input_ids.shape[1] > usable_limit:
            input_ids = input_ids[:, -usable_limit:]
    return input_ids.to(device)


def prepare_output_dir(root: Path):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    metric_fields = [
        "baseline_large_tps", "baseline_large_elapsed_ms", "avg_target_baseline_generate_ms_per_token",
        "speculative_tps", "speedup", "acceptance_rate", "all_accept_round_rate",
        "resample_round_rate", "avg_accepted_per_round", "avg_contributed_per_round",
        "avg_draft_generate_ms_per_round", "avg_target_verify_ms_per_round",
        "avg_draft_prefill_forward_ms_per_round", "avg_draft_decode_forward_ms_per_round",
        "avg_incremental_prefill_forward_ms_per_round",
        "draft_generate_ms_total", "target_verify_ms_total",
        "draft_prefill_forward_ms_total", "draft_decode_forward_ms_total",
        "incremental_prefill_forward_ms_total", "rounds", "generated_tokens"
    ]

    summary = []
    for key, items in grouped.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["trials"] = len(items)
        for field in metric_fields:
            values = [float(item[field]) for item in items]
            row[field] = sum(values) / len(values)
        summary.append(row)

    summary.sort(key=lambda x: (-x["speedup"], x[group_keys[0]]))
    return summary


def maybe_run_warmup(prompts, tokenizer, small_model, large_model, args):
    for prompt_rec in prompts[:args.warmup]:
        reserve_tokens = min(8, args.max_tokens) + max(args.gammas) + 1
        input_ids = encode_prompt(
            tokenizer,
            prompt_rec["prompt"],
            model_device(large_model),
            small_model,
            large_model,
            reserve_tokens=reserve_tokens,
        )
        torch.manual_seed(args.seed)
        _ = autoregressive_sampling(input_ids.clone(), large_model, min(8, args.max_tokens), temperature=args.temperature, top_k=args.top_ks[0], top_p=args.top_ps[0])
        torch.manual_seed(args.seed)
        _ = speculative_sampling_with_stats(input_ids.clone(), small_model, large_model, min(8, args.max_tokens), gamma=args.gammas[0], temperature=args.temperature, top_k=args.top_ks[0], top_p=args.top_ps[0], random_seed=args.seed)
        sync_if_needed(args.device)


def run_trial(prompt_text, prompt_task, prompt_id, tokenizer, small_model, large_model, pair, args, gamma, top_k, top_p):
    reserve_tokens = args.max_tokens + max(args.gammas) + 1
    input_ids = encode_prompt(
        tokenizer,
        prompt_text,
        model_device(large_model),
        small_model,
        large_model,
        reserve_tokens=reserve_tokens,
    )
    torch.manual_seed(args.seed)
    baseline_large_output, baseline_large_elapsed = timed_segment(args.device, autoregressive_sampling, input_ids.clone(), large_model, args.max_tokens, temperature=args.temperature, top_k=top_k, top_p=top_p)

    small_tps = None
    if args.run_small_baseline:
        torch.manual_seed(args.seed)
        _, small_elapsed = timed_segment(args.device, autoregressive_sampling, input_ids.clone(), small_model, args.max_tokens, temperature=args.temperature, top_k=top_k, top_p=top_p)
        small_tps = (baseline_large_output.shape[1] - input_ids.shape[1]) / small_elapsed

    torch.manual_seed(args.seed)
    spec_output, spec_elapsed = timed_segment(args.device, speculative_sampling_with_stats, input_ids.clone(), small_model, large_model, args.max_tokens, gamma=gamma, temperature=args.temperature, top_k=top_k, top_p=top_p, random_seed=args.seed)
    spec_tokens, spec_stats = spec_output
    generated_tokens = int(spec_tokens.shape[1] - input_ids.shape[1])
    baseline_large_tps = generated_tokens / baseline_large_elapsed
    baseline_large_elapsed_ms = baseline_large_elapsed * 1000.0
    avg_target_baseline_generate_ms_per_token = baseline_large_elapsed_ms / generated_tokens if generated_tokens else 0.0
    speculative_tps = generated_tokens / spec_elapsed
    speedup = speculative_tps / baseline_large_tps if baseline_large_tps else 0.0

    return {
        "pair_name": pair["name"],
        "family": pair.get("family", ""),
        "approx_model_name": pair["approx_model_name"],
        "target_model_name": pair["target_model_name"],
        "task": prompt_task,
        "prompt_id": prompt_id,
        "gamma": gamma,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "approx_quant": args.approx_quant,
        "target_quant": args.target_quant,
        "compile_draft": bool(args.compile_draft),
        "compile_mode": args.compile_mode,
        "baseline_large_tps": round(baseline_large_tps, 6),
        "baseline_large_elapsed_ms": round(baseline_large_elapsed_ms, 6),
        "avg_target_baseline_generate_ms_per_token": round(avg_target_baseline_generate_ms_per_token, 6),
        "baseline_small_tps": round(small_tps, 6) if small_tps is not None else "",
        "speculative_tps": round(speculative_tps, 6),
        "speedup": round(speedup, 6),
        "rounds": spec_stats["rounds"],
        "generated_tokens": generated_tokens,
        "accepted_count": spec_stats["accepted_count"],
        "resample_count": spec_stats["resample_count"],
        "target_sample_count": spec_stats["target_sample_count"],
        "proposed_draft_tokens": spec_stats["proposed_draft_tokens"],
        "acceptance_rate": round(spec_stats["acceptance_rate"], 6),
        "all_accept_round_rate": round(spec_stats["all_accept_round_rate"], 6),
        "resample_round_rate": round(spec_stats["resample_round_rate"], 6),
        "avg_accepted_per_round": round(spec_stats["avg_accepted_per_round"], 6),
        "avg_contributed_per_round": round(spec_stats["avg_contributed_per_round"], 6),
        "avg_draft_generate_ms_per_round": round(spec_stats["avg_draft_generate_ms_per_round"], 6),
        "avg_target_verify_ms_per_round": round(spec_stats["avg_target_verify_ms_per_round"], 6),
        "avg_draft_prefill_forward_ms_per_round": round(spec_stats["avg_draft_prefill_forward_ms_per_round"], 6),
        "avg_draft_decode_forward_ms_per_round": round(spec_stats["avg_draft_decode_forward_ms_per_round"], 6),
        "avg_incremental_prefill_forward_ms_per_round": round(spec_stats["avg_incremental_prefill_forward_ms_per_round"], 6),
        "draft_generate_ms_total": round(spec_stats["draft_generate_ms_total"], 6),
        "target_verify_ms_total": round(spec_stats["target_verify_ms_total"], 6),
        "draft_prefill_forward_ms_total": round(spec_stats["draft_prefill_forward_ms_total"], 6),
        "draft_decode_forward_ms_total": round(spec_stats["draft_decode_forward_ms_total"], 6),
        "incremental_prefill_forward_ms_total": round(spec_stats["incremental_prefill_forward_ms_total"], 6),
    }

def run_benchmark(args):
    out_dir = prepare_output_dir(args.output_root)
    dtype = resolve_dtype(args.dtype)
    pairs = select_pairs(load_pairs(args.pairs_file), args.pair_names, args.pair_tiers)
    prompts = load_prompts(args)
    if not pairs:
        raise SystemExit("No model pairs selected.")
    if not prompts:
        raise SystemExit("No prompts selected.")

    (out_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "selected_pairs": pairs, "prompt_count": len(prompts)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8-sig",
    )

    rows = []
    print(f"[run] output_dir={out_dir}")
    print(f"[run] pairs={len(pairs)} prompts={len(prompts)} grid={len(args.gammas) * len(args.top_ks) * len(args.top_ps)}")

    for pair in pairs:
        print(f"[pair] loading approx={pair['approx_model_name']} target={pair['target_model_name']}")
        tokenizer_source = pair.get("tokenizer_name", pair["approx_model_name"])
        tokenizer = load_tokenizer(tokenizer_source, args.trust_remote_code)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        small_model = load_model(
            pair["approx_model_name"],
            dtype,
            args.device,
            args.trust_remote_code,
            args.approx_quant,
            compile_model=args.compile_draft,
            compile_mode=args.compile_mode,
        )
        large_model = load_model(pair["target_model_name"], dtype, args.device, args.trust_remote_code, args.target_quant)
        small_vocab = output_vocab_size(small_model)
        large_vocab = output_vocab_size(large_model)

        if (small_vocab is not None) and (large_vocab is not None) and (small_vocab > large_vocab):
            msg = (
                f"[pair.skip] pair={pair['name']} incompatible vocab size ordering: "
                f"approx={small_vocab}, target={large_vocab}"
            )
            if getattr(args, "fail_on_incompatible_pairs", False):
                raise RuntimeError(msg)
            print(msg)
            del small_model
            del large_model
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            continue
        if (small_vocab is not None) and (large_vocab is not None) and (small_vocab != large_vocab):
            print(
                f"[pair.warn] pair={pair['name']} vocab differs but will run with shared-prefix token ids: "
                f"approx={small_vocab}, target={large_vocab}"
            )

        maybe_run_warmup(prompts, tokenizer, small_model, large_model, args)

        try:
            for prompt_index, prompt_rec in enumerate(prompts, start=1):
                prompt_id = prompt_rec.get("prompt_id", f"{prompt_rec['task']}_{prompt_index}")
                for gamma, top_k, top_p in itertools.product(args.gammas, args.top_ks, args.top_ps):
                    print(f"[trial] pair={pair['name']} task={prompt_rec['task']} prompt={prompt_id} gamma={gamma} top_k={top_k} top_p={top_p}")
                    rows.append(run_trial(prompt_rec["prompt"], prompt_rec["task"], prompt_id, tokenizer, small_model, large_model, pair, args, gamma, top_k, top_p))
        except Exception as e:
            msg = f"[pair.error] pair={pair['name']} error={type(e).__name__}: {e}"
            if getattr(args, "fail_on_pair_error", False):
                raise
            print(msg)
        finally:
            del small_model
            del large_model
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass

    if not rows:
        raise SystemExit("No successful trials. Check pair compatibility or use --fail_on_incompatible_pairs for strict failure.")

    with (out_dir / "trial_results.jsonl").open("w", encoding="utf-8-sig") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(out_dir / "trial_results.csv", rows, list(rows[0].keys()))
    summary_by_config = aggregate_rows(rows, ["pair_name", "family", "approx_quant", "target_quant", "compile_draft", "compile_mode", "gamma", "top_k", "top_p"])
    write_csv(out_dir / "summary_by_config.csv", summary_by_config, list(summary_by_config[0].keys()))
    summary_by_pair = aggregate_rows(rows, ["pair_name", "family", "approx_quant", "target_quant", "compile_draft", "compile_mode"])
    write_csv(out_dir / "summary_by_pair.csv", summary_by_pair, list(summary_by_pair[0].keys()))
    summary_by_task = aggregate_rows(rows, ["pair_name", "family", "approx_quant", "target_quant", "compile_draft", "compile_mode", "task"])
    write_csv(out_dir / "summary_by_task.csv", summary_by_task, list(summary_by_task[0].keys()))

    best = max(summary_by_config, key=lambda x: x["speedup"])
    best_text = [
        f"output_dir={out_dir}",
        f"trial_count={len(rows)}",
        f"best_pair={best['pair_name']}",
        f"best_gamma={best['gamma']}",
        f"best_top_k={best['top_k']}",
        f"best_top_p={best['top_p']}",
        f"best_speedup={best['speedup']:.4f}",
        f"best_acceptance_rate={best['acceptance_rate']:.4f}",
        f"best_speculative_tps={best['speculative_tps']:.4f}",
        f"best_baseline_large_tps={best['baseline_large_tps']:.4f}",
    ]
    (out_dir / "best_result.txt").write_text("\n".join(best_text), encoding="utf-8-sig")
    print("[done] summaries written:")
    print(f"  - {out_dir / 'trial_results.csv'}")
    print(f"  - {out_dir / 'summary_by_config.csv'}")
    print(f"  - {out_dir / 'summary_by_pair.csv'}")
    print(f"  - {out_dir / 'summary_by_task.csv'}")
    print(f"  - {out_dir / 'best_result.txt'}")

