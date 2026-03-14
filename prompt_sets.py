PROMPT_SETS = {
    "continuation": [
        "Continue this technical note in the same style for about one paragraph:\n\nDistributed speculative decoding uses a draft model to propose several tokens and a target model to verify them. Once the dense draft distribution transfer was replaced with sparse transmission, communication overhead dropped sharply. The next bottleneck moved to the target-side computation because",
        "Continue this engineering log in a coherent way:\n\nThe team benchmarked multiple draft-target model pairs on the same GPU. Early results showed that tokenizer alignment mattered more than raw parameter count. When the draft model was too weak, rejection frequency increased and the system lost throughput. A more practical selection rule is",
        "Continue this architecture description:\n\nIn the current design, the edge device runs the draft model and keeps a local KV cache. The server runs the target model, verifies the draft tokens, and returns the committed prefix extension. After switching away from per-round HTTP request-response overhead, the remaining optimization target became",
        "Continue this report section:\n\nFor a fair speedup measurement, speculative decoding must be compared against the same target model running ordinary autoregressive decoding under the same prompt set, max token limit, and sampling parameters. Otherwise the comparison can hide the real cause of slowdown, which is usually"
    ],
    "code_completion": [
        "Complete the Python function:\n\n```python\ndef summarize_trials(rows):\n    \"\"\"rows: list of dicts with keys pair_name, speedup, acceptance_rate.\n    Return a dict {pair_name: {\"mean_speedup\": ..., \"mean_acceptance\": ...}}.\n    \"\"\"\n```",
        "Complete the function body:\n\n```python\ndef select_best_config(records):\n    \"\"\"Return the config dict with the highest speedup.\n    Break ties by higher acceptance_rate, then lower rounds.\n    \"\"\"\n```",
        "Continue this script so it writes a CSV summary:\n\n```python\nimport csv\n\ndef export_summary(path, rows):\n    fieldnames = [\"pair_name\", \"gamma\", \"top_k\", \"top_p\", \"speedup\"]\n```",
        "Complete the utility function:\n\n```python\ndef acceptance_rate(accepted_count, rounds, gamma):\n    \"\"\"Return accepted draft tokens divided by proposed draft tokens.\"\"\"\n```"
    ],
    "long_summarization": [
        "Summarize the following into five concise bullet points:\n\nA distributed speculative decoding system was built by separating the draft model and target model across two machines. The server handled accept, reject, and resample decisions while both sides maintained KV cache. Initial performance was poor because the device transmitted full q probability tensors every round, producing payloads around one megabyte per step. After replacing dense q transmission with a sparse representation, request size dropped to around ten kilobytes and communication latency fell sharply. However, total throughput still depended heavily on acceptance rate and server-side target inference time. Later experiments replaced HTTP with UDP for functional testing, which reduced communication overhead further and exposed server compute as the dominant bottleneck.",
        "Write a short summary of this passage:\n\nThe engineering team needed a fair way to measure speedup. They defined baseline as remote autoregressive decoding with the target model under the same prompt set and sampling parameters. They rejected comparisons against unrelated single-machine runs because those mixed communication cost, algorithmic benefit, and hardware effects. They also found that raw RTT measurements from tiny TCP packets underestimated the cost of HTTP request-response chains. Once the protocol changed to UDP for testing, measured communication latency dropped to the low millisecond range, confirming that network transport itself was not the main issue.",
        "Summarize this report section in one paragraph:\n\nJetson Nano was later used as the draft-side device. Deployment required careful compatibility work because the board ran an older Python and transformers stack than the desktop server. The team reused the same speculative decoding semantics but adapted some imports and model-loading calls for older libraries. After these changes, the Jetson device successfully participated in the distributed workflow. Throughput was lower than desktop-only runs, but the architecture was validated and the next phase shifted toward finding better model pairs.",
        "Condense the following into three bullets:\n\nChoosing a good model pair for speculative decoding is not only about making the target larger. If the target distribution diverges too far from the draft model, rejection and resampling can dominate. If the draft is too large, the speculative path loses its speed advantage. The most promising pairs usually share tokenizer family, remain close enough in distribution for reasonable acceptance, and keep the draft small enough for edge deployment."
    ],
    "structured_extraction": [
        "Read the text and output JSON with keys issue, cause, fix, metric:\n\nIssue: throughput dropped below baseline. Cause: acceptance rate was too low and per-round request overhead was high. Fix: sparse q transmission and lower-overhead communication. Metric: speedup.",
        "Extract a JSON object with keys role_device, role_server, protocol, bottleneck:\n\nThe device runs the draft model, while the server runs the target model and performs accept/reject/resample. The latest prototype uses UDP for testing and now spends most of its time in server-side target computation.",
        "Convert the following note to JSON with keys model_pair, gamma, top_k, top_p, goal:\n\nTry distilgpt2 as draft and gpt2-medium as target with gamma 4, top_k 50, top_p 0.9 to see whether acceptance improves without making the draft too heavy for future Jetson deployment.",
        "Extract fields as compact JSON:\n\nSystem: distributed speculative decoding. Baseline: remote target autoregressive decoding. Main metric: speedup. Secondary metrics: acceptance rate, rounds, average contributed tokens per round."
    ],
    "instruction_following": [
        "Give a concise experimental plan for comparing three speculative decoding model pairs under the same hardware and prompt conditions.",
        "Propose three ways to increase acceptance rate without changing the core accept/reject/resample rule.",
        "Explain to an engineer why model pairs from the same tokenizer family often behave better in speculative decoding.",
        "List four concrete signals that a speculative decoding setup is bottlenecked by server compute rather than communication."
    ]
}

TASK_GROUPS = {
    "core": ["continuation", "code_completion", "long_summarization"],
    "extended": ["structured_extraction", "instruction_following"],
    "all": list(PROMPT_SETS.keys())
}

def normalize_names(names):
    if not names:
        return list(PROMPT_SETS.keys())
    expanded = []
    for name in names:
        if name in TASK_GROUPS:
            expanded.extend(TASK_GROUPS[name])
        else:
            expanded.append(name)
    seen = set()
    ordered = []
    for name in expanded:
        if name in PROMPT_SETS and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered

def get_prompt_sets(names=None, limit_per_set=None):
    names = normalize_names(names)
    selected = {}
    for name in names:
        prompts = PROMPT_SETS[name]
        if limit_per_set is not None:
            prompts = prompts[:limit_per_set]
        selected[name] = prompts
    return selected
