# RECALL
本仓库为RECALL 端到端在线流程。

## 快速入口

- 运行论文流程（在线维护图 + 触发 + 双路径检索 + TextPack/GraphPack + LLM）：`scripts/recall_run.py`

## 数据（Dataset）

- LogHub（LogPai）：https://github.com/logpai/loghub
- USENIX CFDR（HPC4）：https://www.usenix.org/cfdr-data#hpc4

## 本地 LLM 分支（语义实体抽取）

论文语义通道默认可用在线 LLM；本仓库也支持本地 HuggingFace 模型分支：

```bash
python scripts/recall_run.py \
  --dataset bgl \
  --loghub_root /path/to/LogHub \
  --output_dir out/recall_bgl \
  --no_llm \
  --enable_semantic_channel \
  --semantic_backend local_hf \
  --semantic_model_path /path/to/local/hf/model
```

## 目录结构（简述）

- `recall/`：论文实现（实体抽取、动态图维护、触发、双路径检索、打包、prompt、LLM）
- `scripts/recall_run.py`：运行入口（生成 predictions.jsonl + metrics.json）
- `tests/`：核心组件的单元测试
