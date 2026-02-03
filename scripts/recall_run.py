from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recall.config import RecallConfig
from recall.data import load_dataset
from recall.pipeline import RecallPipeline

def main() -> None:
    p = argparse.ArgumentParser(description='Run RECALL paper-aligned online pipeline (Algorithm 1/2).')
    p.add_argument('--dataset', required=True, choices=['bgl', 'thunderbird', 'tdb', 'spirit'], help='Dataset name')
    p.add_argument('--loghub_root', default=None, help='Path to LogHub folder (default: ../LogHub/)')
    p.add_argument('--output_dir', default='out/recall', help='Output directory')
    p.add_argument('--max_logs', type=int, default=None, help='Optional cap for number of logs to process')
    p.add_argument('--api_key', default=None, help='DeepSeek API key (or set DEEPSEEK_API_KEY env)')
    p.add_argument('--api_key_file', default=None, help='Path to file containing API key')
    p.add_argument('--no_llm', action='store_true', help='Disable LLM inference (offline run)')
    p.add_argument('--enable_semantic_channel', action='store_true', help='Enable semantic entity extraction channel')
    p.add_argument('--semantic_backend', choices=['deepseek_api', 'local_hf'], default='deepseek_api', help='Semantic channel backend')
    p.add_argument('--semantic_model_path', default=None, help='Local HF model path for semantic channel (when semantic_backend=local_hf)')
    p.add_argument('--llm_backend', choices=['deepseek_api', 'local_hf'], default='deepseek_api', help='Decision LLM backend')
    p.add_argument('--llm_model_path', default=None, help='Local HF model path for decision LLM (when llm_backend=local_hf)')
    p.add_argument('--theta_tc', type=int, default=2)
    p.add_argument('--theta_rf', type=int, default=2)
    p.add_argument('--delta_t_sec', type=int, default=300)
    p.add_argument('--graph_window_t_sec', type=int, default=900)
    p.add_argument('--temporal_k', type=int, default=15)
    p.add_argument('--evidence_budget_nmax', type=int, default=30)
    p.add_argument('--degree_threshold_dmax', type=int, default=200)
    p.add_argument('--decay_lambda', type=float, default=None)
    p.add_argument('--theta_w', type=float, default=0.05)
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RecallConfig(theta_tc=int(args.theta_tc), theta_rf=int(args.theta_rf), delta_t_sec=int(args.delta_t_sec), graph_window_t_sec=int(args.graph_window_t_sec), temporal_k=int(args.temporal_k), evidence_budget_nmax=int(args.evidence_budget_nmax), degree_threshold_dmax=int(args.degree_threshold_dmax), decay_lambda=None if args.decay_lambda is None else float(args.decay_lambda), theta_w=float(args.theta_w), enable_semantic_channel=bool(args.enable_semantic_channel), semantic_backend=str(args.semantic_backend), semantic_local_model_path=args.semantic_model_path, llm_backend=str(args.llm_backend), llm_local_model_path=args.llm_model_path)
    records = load_dataset(args.dataset, loghub_root=args.loghub_root)
    pipe = RecallPipeline(cfg=cfg, api_key=args.api_key, api_key_file=args.api_key_file, enable_llm=not args.no_llm)
    outputs = pipe.process(records, max_logs=args.max_logs)
    preds_path = out_dir / 'predictions.jsonl'
    with open(preds_path, 'w', encoding='utf-8') as w:
        for rec in outputs.predictions:
            w.write(json.dumps(rec, ensure_ascii=False) + '\n')
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as w:
        json.dump(outputs.metrics, w, ensure_ascii=False, indent=2)
    print(f'✅ Done. Wrote {len(outputs.predictions):,} predictions')
    print(f'   Predictions: {preds_path}')
    print(f'   Metrics:     {metrics_path}')
    print(f"   F1={outputs.metrics.get('f1'):.4f} P={outputs.metrics.get('precision'):.4f} R={outputs.metrics.get('recall'):.4f}")
if __name__ == '__main__':
    main()
