from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional
from .config import RecallConfig
from .data import LogRecord
from .dynamic_graph import DynamicLogEntityGraph
from .entity_extraction import SemanticEntityExtractor, StatisticalEntityExtractor, extract_entities
from .llm_client import DeepSeekClient, LocalHfClient, parse_decision
from .metrics import compute_metrics
from .packaging import build_evidence_pack
from .prompt import build_prompt
from .retrieval import dual_path_retrieve
from .trigger import TriggerEngine, severity_level

@dataclass
class PipelineOutputs:
    predictions: List[Dict]
    metrics: Dict

class RecallPipeline:

    def __init__(self, cfg: RecallConfig, api_key: Optional[str]=None, api_key_file: Optional[str]=None, enable_llm: bool=True) -> None:
        self.cfg = cfg
        self.graph = DynamicLogEntityGraph(cfg)
        self.stat_extractor = StatisticalEntityExtractor(cfg)
        self.semantic_extractor = None
        if cfg.enable_semantic_channel:
            self.semantic_extractor = SemanticEntityExtractor(cfg, backend=cfg.semantic_backend, api_key=api_key, api_key_file=api_key_file, local_model_path=cfg.semantic_local_model_path)
        self.trigger = TriggerEngine(cfg)
        self.enable_llm = bool(enable_llm)
        self.llm = None
        if self.enable_llm:
            b = (cfg.llm_backend or '').strip().lower()
            if b in ('deepseek', 'deepseek_api', 'api'):
                if not (api_key or api_key_file):
                    raise ValueError('LLM inference enabled but no API key provided')
                self.llm = DeepSeekClient(api_key=api_key, api_key_file=api_key_file, endpoint=cfg.llm_endpoint, model=cfg.llm_model_name)
            elif b in ('local', 'local_hf', 'hf'):
                if not cfg.llm_local_model_path:
                    raise ValueError('local llm backend requires --llm_model_path')
                self.llm = LocalHfClient(model_path=cfg.llm_local_model_path, max_new_tokens=256)
            else:
                raise ValueError(f'Unknown llm backend: {cfg.llm_backend}')

    def process(self, records: List[LogRecord], max_logs: Optional[int]=None) -> PipelineOutputs:
        preds: List[Dict] = []
        true_labels: List[int] = []
        pred_labels: List[int] = []
        for (i, rec) in enumerate(records):
            if max_logs is not None and i >= int(max_logs):
                break
            ts = int(rec.ts_sec)
            msg = rec.message or ''
            sev = severity_level(self.cfg, msg)
            ent_res = extract_entities(cfg=self.cfg, stat_extractor=self.stat_extractor, ts_sec=ts, message=msg, semantic_extractor=self.semantic_extractor)
            self.graph.add_log(log_id=int(rec.log_id), ts_sec=ts, message=msg, entities=ent_res.final, severity=sev)
            self.graph.tick(ts)
            trig = self.trigger.check(ts, msg)
            decision = {'label': 'NORMAL', 'confidence': 0.0, 'evidence_ids': [], 'rationale': ''}
            prompt_text = ''
            evidence_items = []
            if trig.triggered:
                evidence_items = dual_path_retrieve(self.cfg, self.graph, int(rec.log_id))
                pack = build_evidence_pack(self.cfg, self.graph, int(rec.log_id), evidence_items)
                pb = build_prompt(self.graph, int(rec.log_id), pack)
                prompt_text = pb.prompt
                if self.enable_llm and self.llm is not None:
                    raw = self.llm.chat(prompt_text)
                    d = parse_decision(raw)
                    decision = {'label': d.label, 'confidence': d.confidence, 'evidence_ids': d.evidence_ids, 'rationale': d.rationale, 'llm_error': d.error, 'llm_raw': d.raw}
                else:
                    decision['confidence'] = 0.0
            pred_label_int = 1 if decision['label'] == 'ANOMALY' else 0
            out = {'log_id': int(rec.log_id), 'timestamp': int(ts), 'message': msg, 'true_label': int(rec.true_label), 'triggered': bool(trig.triggered), 'trigger_by': trig.by, 'severity': int(sev), 'entities_stat': sorted(list(ent_res.estat)), 'entities_stat_validated': sorted(list(ent_res.estat_validated)), 'entities_sem': sorted(list(ent_res.esem)), 'entities_final': sorted(list(ent_res.final)), 'prediction': decision}
            if trig.triggered:
                out['retrieval'] = {'evidence_count': len(evidence_items), 'evidence_log_ids': [int(e.log_id) for e in evidence_items]}
                out['prompt_len'] = len(prompt_text)
            preds.append(out)
            true_labels.append(int(rec.true_label))
            pred_labels.append(int(pred_label_int))
        m = compute_metrics(true_labels, pred_labels)
        return PipelineOutputs(predictions=preds, metrics=m.as_dict())
