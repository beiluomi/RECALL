from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from .config import RecallConfig
from .dynamic_graph import DynamicLogEntityGraph
from .text import normalize_message_for_dedup
from .trigger import severity_level

@dataclass
class EvidenceItem:
    log_id: int
    ts_sec: int
    message: str
    severity: int
    dist: int
    score: float
    edge_weight: float
    paths: Set[str] = field(default_factory=set)
    shared_entities: List[str] = field(default_factory=list)
    time_offset: Optional[int] = None

def _min_dist(a: Optional[int], b: Optional[int]) -> int:
    if a is None:
        return int(b) if b is not None else 999999
    if b is None:
        return int(a)
    return int(min(a, b))

def _temporal_path_min_edge_weight(g: DynamicLogEntityGraph, src: int, dst: int, now_ts: int) -> float:
    if src == dst:
        return 1.0
    w_min = 1.0
    cur = dst
    visited = 0
    while cur != src and visited < 2048:
        ln = g.get_log(cur)
        if ln is None or ln.prev_log_id is None:
            break
        prev = ln.prev_log_id
        w = g.temporal_edge_weight(prev, cur, now_ts)
        w_min = min(w_min, w)
        cur = prev
        visited += 1
    if cur == src:
        return w_min
    w_min = 1.0
    cur = src
    visited = 0
    while cur != dst and visited < 2048:
        ln = g.get_log(cur)
        if ln is None or ln.next_log_id is None:
            break
        nxt = ln.next_log_id
        w = g.temporal_edge_weight(cur, nxt, now_ts)
        w_min = min(w_min, w)
        cur = nxt
        visited += 1
    if cur == dst:
        return w_min
    return 0.0

def dual_path_retrieve(cfg: RecallConfig, g: DynamicLogEntityGraph, target_log_id: int) -> List[EvidenceItem]:
    tgt = g.get_log(target_log_id)
    if tgt is None:
        return []
    now_ts = int(tgt.ts_sec)
    eq = g.get_entities_for_log(target_log_id)
    cand_dist_time: Dict[int, int] = {}
    cand_dist_struct: Dict[int, int] = {}
    shared_entities: Dict[int, Set[str]] = {}
    for e in eq:
        if g.entity_degree(e) > int(cfg.degree_threshold_dmax):
            continue
        for lid in g.get_logs_for_entity(e):
            if lid == target_log_id:
                continue
            cand_dist_struct[lid] = _min_dist(cand_dist_struct.get(lid), 2)
            shared_entities.setdefault(lid, set()).add(e)
    k = int(cfg.temporal_k)
    cur = target_log_id
    for d in range(1, k + 1):
        ln = g.get_log(cur)
        if ln is None or ln.prev_log_id is None:
            break
        cur = ln.prev_log_id
        cand_dist_time[cur] = min(cand_dist_time.get(cur, 10 ** 9), d)
    cur = target_log_id
    for d in range(1, k + 1):
        ln = g.get_log(cur)
        if ln is None or ln.next_log_id is None:
            break
        cur = ln.next_log_id
        cand_dist_time[cur] = min(cand_dist_time.get(cur, 10 ** 9), d)
    cand_ids = set(cand_dist_struct.keys()) | set(cand_dist_time.keys())
    msg2best: Dict[str, int] = {}
    for lid in cand_ids:
        ln = g.get_log(lid)
        if ln is None:
            continue
        key = normalize_message_for_dedup(ln.message, case_insensitive=cfg.dedup_case_insensitive)
        if not key:
            continue
        prev = msg2best.get(key)
        if prev is None:
            msg2best[key] = lid
        else:
            a = g.get_log(prev)
            if a is None or ln.ts_sec > a.ts_sec:
                msg2best[key] = lid
    dedup_ids = set(msg2best.values())
    items: List[EvidenceItem] = []
    for lid in dedup_ids:
        ln = g.get_log(lid)
        if ln is None:
            continue
        dist = _min_dist(cand_dist_struct.get(lid), cand_dist_time.get(lid))
        dist = max(1, int(dist))
        sev = int(ln.severity)
        if sev == 0:
            sev = severity_level(cfg, ln.message)
        w = 0.0
        if lid in cand_dist_time:
            w = max(w, _temporal_path_min_edge_weight(g, target_log_id, lid, now_ts))
        if lid in cand_dist_struct:
            best = 0.0
            for e in shared_entities.get(lid, set()):
                w0 = g.structural_edge_weight(target_log_id, e, now_ts)
                wi = g.structural_edge_weight(lid, e, now_ts)
                best = max(best, min(w0, wi))
            w = max(w, best)
        score = cfg.score_a * float(sev) + cfg.score_b * (1.0 / float(dist)) + cfg.score_c * float(w)
        paths: Set[str] = set()
        if lid in cand_dist_struct:
            paths.add('struct')
        if lid in cand_dist_time:
            paths.add('time')
        item = EvidenceItem(log_id=lid, ts_sec=int(ln.ts_sec), message=ln.message, severity=int(sev), dist=int(dist), score=float(score), edge_weight=float(w), paths=paths, shared_entities=sorted(list(shared_entities.get(lid, set()))))
        if lid in cand_dist_time:
            item.time_offset = int(ln.ts_sec) - int(tgt.ts_sec)
        items.append(item)
    items.sort(key=lambda x: (x.score, x.ts_sec), reverse=True)
    return items[:int(cfg.evidence_budget_nmax)]
