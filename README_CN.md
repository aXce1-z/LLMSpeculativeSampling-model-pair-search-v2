# LLMSpeculativeSampling Model Pair Search V2

这个目录是新版独立评测工具：

- 不修改 `E:\LLMSpeculativeSampling-main`
- 不依赖原项目旧的 `sampling/kvcache_model.py`
- 面向 `llmsd_v2` 新环境
- 优先支持 Qwen2.5 / Llama 3.2 / OPT / Pythia / GPT2

## 推荐环境

- Python 3.10
- torch 2.5.1+cu121
- transformers 4.46.3
- accelerate 1.13.0
- numpy 1.26.4

## 第一轮推荐命令

```powershell
conda activate llmsd_v2
$env:HF_HOME="E:\hf_cache"
$env:HUGGINGFACE_HUB_CACHE="E:\hf_cache\hub"
$env:TRANSFORMERS_CACHE="E:\hf_cache\transformers"
New-Item -ItemType Directory -Force E:\hf_cache\hub | Out-Null
New-Item -ItemType Directory -Force E:\hf_cache\transformers | Out-Null
cd E:\LLMSpeculativeSampling-model-pair-search-v2
python run_model_pair_search.py `
  --pair_names qwen25_0p5b_to_1p5b qwen25_0p5b_to_3b qwen25_1p5b_to_3b distilgpt2_to_gpt2_medium `
  --prompt_sets core `
  --prompts_per_set 2 `
  --gammas 2 4 `
  --top_ks 20 50 `
  --top_ps 0 0.9 `
  --max_tokens 64
```

## 第二轮（如果已有 Llama 访问权限）

```powershell
conda activate llmsd_v2
$env:HF_HOME="E:\hf_cache"
$env:HUGGINGFACE_HUB_CACHE="E:\hf_cache\hub"
$env:TRANSFORMERS_CACHE="E:\hf_cache\transformers"
cd E:\LLMSpeculativeSampling-model-pair-search-v2
python run_model_pair_search.py `
  --pair_names llama32_1b_to_3b `
  --prompt_sets core `
  --prompts_per_set 2 `
  --gammas 2 4 `
  --top_ks 20 50 `
  --top_ps 0 0.9 `
  --max_tokens 64
```

## 输出文件

每次运行会生成：

- `outputs/run_YYYYMMDD_HHMMSS/run_config.json`
- `outputs/run_YYYYMMDD_HHMMSS/trial_results.jsonl`
- `outputs/run_YYYYMMDD_HHMMSS/trial_results.csv`
- `outputs/run_YYYYMMDD_HHMMSS/summary_by_config.csv`
- `outputs/run_YYYYMMDD_HHMMSS/summary_by_pair.csv`
- `outputs/run_YYYYMMDD_HHMMSS/best_result.txt`

## 时间统计

每条 trial 包含：

- `avg_draft_generate_ms_per_round`
- `avg_target_verify_ms_per_round`
- `draft_generate_ms_total`
- `target_verify_ms_total`

## Colab

这个目录是自包含的。把整个目录上传到 Colab 后直接运行，不需要再依赖原项目的 `sampling/` 目录。
