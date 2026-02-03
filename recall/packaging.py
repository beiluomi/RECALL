from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .config import RecallConfig
from .dynamic_graph import DynamicLogEntityGraph
from .entity_extraction import classify_entity_type
from .retrieval import EvidenceItem

@dataclass
class EvidencePack:
    textpack: str
    graphpack_json: str
    id_map_logs: Dict[int, str]
    id_map_entities: Dict[str, str]

def _ts(ts_sec: int) -> str:
    return str(int(ts_sec))

def build_evidence_pack(cfg: RecallConfig, g: DynamicLogEntityGraph, target_log_id: int, evidence: List[EvidenceItem]) -> EvidencePack:
    tgt = g.get_log(target_log_id)
    if tgt is None:
        raise ValueError(f'target_log_id not found: {target_log_id}')
    id_map_logs: Dict[int, str] = {target_log_id: 'L0'}
    for (i, it) in enumerate(evidence, start=1):
        id_map_logs[it.log_id] = f'L{i}'
    ent_set = set()
    for lid in id_map_logs.keys():
        for e in g.get_entities_for_log(lid):
            ent_set.add(e)
    id_map_entities: Dict[str, str] = {}
    for (i, e) in enumerate(sorted(ent_set), start=1):
        id_map_entities[e] = f'E{i}'
    lines: List[str] = []
    lines.append('=== TEXT EVIDENCE (TextPack) ===')
    ordered_log_ids = [target_log_id] + [it.log_id for it in evidence]
    text_log_ids = [it.log_id for it in evidence]
    for lid in text_log_ids:
        ln = g.get_log(lid)
        if ln is None:
            continue
        lid_local = id_map_logs[lid]
        lines.append(f'{lid_local}: ts={_ts(ln.ts_sec)} severity={ln.severity} {ln.message}')
    textpack = '\n'.join(lines)
    nodes: List[Dict] = []
    for lid in ordered_log_ids:
        ln = g.get_log(lid)
        if ln is None:
            continue
        nodes.append({'id': id_map_logs[lid], 'type': 'log', 'timestamp': int(ln.ts_sec), 'severity': int(ln.severity)})
    for (e, eid) in id_map_entities.items():
        nodes.append({'id': eid, 'type': 'entity', 'entity_type': classify_entity_type(e), 'value': e})
    edges: List[Dict] = []
    rel_id = 1
    now_ts = int(tgt.ts_sec)
    for lid in ordered_log_ids:
        for e in sorted(g.get_entities_for_log(lid)):
            if e not in id_map_entities:
                continue
            w = g.structural_edge_weight(lid, e, now_ts)
            if w < cfg.theta_w:
                continue
            edges.append({'id': f'R{rel_id}', 'type': 'struct', 'source': id_map_logs[lid], 'target': id_map_entities[e], 'weight': round(float(w), 6)})
            rel_id += 1
    selected_set = set(ordered_log_ids)
    for lid in ordered_log_ids:
        ln = g.get_log(lid)
        if ln is None or ln.next_log_id is None:
            continue
        if ln.next_log_id not in selected_set:
            continue
        w = g.temporal_edge_weight(lid, ln.next_log_id, now_ts)
        if w < cfg.theta_w:
            continue
        edges.append({'id': f'R{rel_id}', 'type': 'time', 'source': id_map_logs[lid], 'target': id_map_logs[ln.next_log_id], 'weight': round(float(w), 6)})
        rel_id += 1
    summary: List[str] = []
    tgt_ents = g.get_entities_for_log(target_log_id)
    for it in evidence:
        shared = sorted(list(tgt_ents.intersection(g.get_entities_for_log(it.log_id))))
        if shared:
            es = ', '.join((id_map_entities[e] for e in shared if e in id_map_entities))
            summary.append(f'{id_map_logs[it.log_id]} shares entities {es} with L0')
        if it.time_offset is not None and abs(int(it.time_offset)) <= cfg.temporal_k:
            summary.append(f'{id_map_logs[it.log_id]} is within K-step temporal context of L0 (offset {it.time_offset}s)')
    graphpack = {'nodes': nodes, 'edges': edges, 'summary': summary[:50]}
    graphpack_json = json.dumps(graphpack, ensure_ascii=False)
    return EvidencePack(textpack=textpack, graphpack_json=graphpack_json, id_map_logs=id_map_logs, id_map_entities=id_map_entities)
