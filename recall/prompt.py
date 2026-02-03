from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .dynamic_graph import DynamicLogEntityGraph
from .packaging import EvidencePack

@dataclass
class PromptBundle:
    prompt: str
    target_log_id: int
    graphpack_json: str
    textpack: str

def build_prompt(g: DynamicLogEntityGraph, target_log_id: int, pack: EvidencePack) -> PromptBundle:
    tgt = g.get_log(target_log_id)
    if tgt is None:
        raise ValueError(f'target_log_id not found: {target_log_id}')
    system = '[Task]\nYou are a senior site reliability engineer. Determine whether the TARGET log indicates an anomaly based on the provided evidence.\n\n[Evidence Presentation]\n- Each log entry has a unique ID (L0 is the target; L1.. are evidence logs).\n- Each entity has a unique ID (E1..).\n- Each relation has a unique ID (R1..).\n\n[Output Constraint]\nReturn a JSON object with exactly these fields:\n- label: "ANOMALY" or "NORMAL"\n- confidence: a number in [0,1]\n- evidence_ids: a list of evidence IDs you relied upon, e.g., ["L1","E2"]\n- rationale: a brief explanation that cites the evidence IDs\n\n[Reasoning Constraint]\nExplain briefly and cite concrete evidence IDs. Do not add any extra keys or any markdown.\n'
    target = f'=== TARGET LOG ===\nLog ID: L0\nTimestamp: {tgt.ts_sec}\nSeverity: {tgt.severity}\nContent: {tgt.message}\n'
    prompt = f'{system}\n\n{target}\n\n{pack.textpack}\n\n=== TOPOLOGICAL EVIDENCE (GraphPack as JSON) ===\n{pack.graphpack_json}\n\n[Output Requirements]\nOutput ONLY valid JSON:\n{{\n  "label": "ANOMALY" or "NORMAL",\n  "confidence": 0.0-1.0,\n  "evidence_ids": ["L1","L2","E1"],\n  "rationale": "..."\n}}'
    return PromptBundle(prompt=prompt, target_log_id=target_log_id, graphpack_json=pack.graphpack_json, textpack=pack.textpack)
