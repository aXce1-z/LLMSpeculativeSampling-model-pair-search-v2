# Benchmark Prompt Builder

这个脚本从公开数据集抽样，生成固定的 `JSONL` benchmark prompt 集。

默认抽样：

- 摘要：20 条
- 代码：20 条
- 数学推理：20 条

总计 60 条。

## 依赖

在 `llmsd_v2` 环境里额外安装：

```powershell
conda activate llmsd_v2
python -m pip install datasets
```

## 生成命令

使用 `cnn_dailymail`：

```powershell
conda activate llmsd_v2
cd E:\LLMSpeculativeSampling-model-pair-search-v2
python build_benchmark_jsonl.py --output E:\LLMSpeculativeSampling-model-pair-search-v2\benchmark_prompts.jsonl --summ_source cnn_dailymail --summ_count 20 --code_count 20 --math_count 20
```

使用 `xsum`：

```powershell
conda activate llmsd_v2
cd E:\LLMSpeculativeSampling-model-pair-search-v2
python build_benchmark_jsonl.py --output E:\LLMSpeculativeSampling-model-pair-search-v2\benchmark_prompts.jsonl --summ_source xsum --summ_count 20 --code_count 20 --math_count 20
```

## 生成后如何跑评测

```powershell
conda activate llmsd_v2
$env:HF_HOME="E:\hf_cache"
$env:HUGGINGFACE_HUB_CACHE="E:\hf_cache\hub"
$env:TRANSFORMERS_CACHE="E:\hf_cache\transformers"
cd E:\LLMSpeculativeSampling-model-pair-search-v2
python run_model_pair_search.py `
  --prompt_file E:\LLMSpeculativeSampling-model-pair-search-v2\benchmark_prompts.jsonl `
  --pair_names qwen25_0p5b_to_1p5b qwen25_0p5b_to_3b qwen25_1p5b_to_3b distilgpt2_to_gpt2_medium `
  --gammas 2 4 `
  --top_ks 20 50 `
  --top_ps 0 0.9 `
  --max_tokens 64
```
