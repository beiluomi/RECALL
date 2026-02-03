from __future__ import annotations
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
import math
from .config import RecallConfig
from .entity_extraction import classify_entity_type

@dataclass
class LogNode:
    log_id: int
    ts_sec: int
    message: str
    severity: int
    prev_log_id: Optional[int] = None
    next_log_id: Optional[int] = None

@dataclass
class EntityNode:
    value: str
    etype: str
    activity: float = 0.0
    last_step: int = 0
    last_seen_ts: int = 0

class DynamicLogEntityGraph:

    def __init__(self, cfg: RecallConfig) -> None:
        self.cfg = cfg
        self._step = 0
        self.logs: Dict[int, LogNode] = {}
        self.entities: Dict[str, EntityNode] = {}
        self.log_to_entities: Dict[int, Set[str]] = {}
        self.entity_to_logs: Dict[str, Deque[int]] = defaultdict(deque)
        self._log_window: Deque[int] = deque()
        self._last_log_id: Optional[int] = None

    def _lambda(self) -> float:
        lam = self.cfg.decay_lambda
        if lam is not None:
            return float(lam)
        T = float(self.cfg.graph_window_t_sec)
        theta_w = float(self.cfg.theta_w)
        if T <= 0 or theta_w <= 0:
            return 0.0
        try:
            return -math.log(theta_w) / T
        except Exception:
            return 0.0

    @property
    def step(self) -> int:
        return self._step

    def _edge_weight(self, edge_last_ts: int, now_ts: int) -> float:
        dt = max(0, int(now_ts) - int(edge_last_ts))
        lam = self._lambda()
        if lam <= 0:
            return 1.0
        return math.exp(-lam * dt)

    def structural_edge_weight(self, log_id: int, entity: str, now_ts: int) -> float:
        ln = self.logs.get(log_id)
        if not ln:
            return 0.0
        return self._edge_weight(ln.ts_sec, now_ts)

    def temporal_edge_weight(self, src_log_id: int, dst_log_id: int, now_ts: int) -> float:
        ln = self.logs.get(dst_log_id)
        if not ln:
            return 0.0
        return self._edge_weight(ln.ts_sec, now_ts)

    def add_log(self, log_id: int, ts_sec: int, message: str, entities: Iterable[str], severity: int) -> None:
        self._step += 1
        ts = int(ts_sec)
        node = LogNode(log_id=log_id, ts_sec=ts, message=message or '', severity=int(severity))
        self.logs[log_id] = node
        self._log_window.append(log_id)
        if self._last_log_id is not None and self._last_log_id in self.logs:
            prev = self.logs[self._last_log_id]
            prev.next_log_id = log_id
            node.prev_log_id = prev.log_id
        self._last_log_id = log_id
        ent_set: Set[str] = set()
        for e in entities:
            e2 = (e or '').strip()
            if not e2:
                continue
            if self.cfg.is_blacklisted_entity(e2):
                continue
            ent_set.add(e2)
        self.log_to_entities[log_id] = ent_set
        for e in ent_set:
            en = self.entities.get(e)
            if en is None:
                en = EntityNode(value=e, etype=classify_entity_type(e), activity=0.0, last_step=self._step, last_seen_ts=ts)
                self.entities[e] = en
            self._activate_entity(en, ts_sec=ts)
            self.entity_to_logs[e].append(log_id)

    def _activate_entity(self, en: EntityNode, ts_sec: int) -> None:
        dt_steps = max(0, self._step - int(en.last_step))
        beta = float(self.cfg.activity_beta)
        if dt_steps > 0:
            en.activity = en.activity * beta ** dt_steps
        en.activity = en.activity + float(self.cfg.activity_alpha)
        en.last_step = self._step
        en.last_seen_ts = int(ts_sec)

    def tick(self, now_ts_sec: int) -> None:
        now = int(now_ts_sec)
        self._evict_logs(now)
        self._prune_edges(now)
        self._prune_entities_by_activity()

    def _evict_logs(self, now_ts: int) -> None:
        if self.cfg.graph_window_t_sec <= 0:
            return
        cutoff = now_ts - int(self.cfg.graph_window_t_sec)
        while self._log_window and self.logs.get(self._log_window[0]) and (self.logs[self._log_window[0]].ts_sec < cutoff):
            lid = self._log_window.popleft()
            self._remove_log(lid)

    def _remove_log(self, log_id: int) -> None:
        ln = self.logs.pop(log_id, None)
        ents = self.log_to_entities.pop(log_id, set())
        if ln is not None:
            if ln.prev_log_id is not None and ln.prev_log_id in self.logs:
                self.logs[ln.prev_log_id].next_log_id = ln.next_log_id
            if ln.next_log_id is not None and ln.next_log_id in self.logs:
                self.logs[ln.next_log_id].prev_log_id = ln.prev_log_id
            if self._last_log_id == log_id:
                self._last_log_id = ln.prev_log_id
        for e in ents:
            dq = self.entity_to_logs.get(e)
            if not dq:
                continue
            try:
                while dq and dq[0] == log_id:
                    dq.popleft()
            except Exception:
                pass
            if dq and log_id in dq:
                self.entity_to_logs[e] = deque([x for x in dq if x != log_id])
            if not self.entity_to_logs[e]:
                self.entity_to_logs.pop(e, None)
                self.entities.pop(e, None)

    def _prune_edges(self, now_ts: int) -> None:
        theta_w = float(self.cfg.theta_w)
        if theta_w <= 0:
            return
        lam = self._lambda()
        if lam <= 0:
            return
        age_limit = int(math.ceil(-math.log(theta_w) / lam))
        if age_limit <= 0:
            return
        cutoff_ts = now_ts - age_limit
        for lid in list(self._log_window):
            ln = self.logs.get(lid)
            if ln is None:
                continue
            if ln.ts_sec >= cutoff_ts:
                continue
            ents = self.log_to_entities.get(lid)
            if not ents:
                continue
            for e in list(ents):
                dq = self.entity_to_logs.get(e)
                if dq:
                    if lid in dq:
                        self.entity_to_logs[e] = deque([x for x in dq if x != lid])
                    if not self.entity_to_logs[e]:
                        self.entity_to_logs.pop(e, None)
                        self.entities.pop(e, None)
            self.log_to_entities[lid] = set()
        for lid in list(self._log_window):
            ln = self.logs.get(lid)
            if ln is None or ln.prev_log_id is None:
                continue
            if ln.ts_sec < cutoff_ts:
                prev = self.logs.get(ln.prev_log_id)
                if prev is not None:
                    prev.next_log_id = None
                ln.prev_log_id = None

    def _entity_activity_now(self, en: EntityNode) -> float:
        dt_steps = max(0, self._step - int(en.last_step))
        beta = float(self.cfg.activity_beta)
        if dt_steps <= 0:
            return float(en.activity)
        return float(en.activity) * beta ** dt_steps

    def _prune_entities_by_activity(self) -> None:
        eps = float(self.cfg.activity_epsilon)
        if eps <= 0:
            return
        if self._step % 256 != 0:
            return
        to_drop: List[str] = []
        for (e, en) in self.entities.items():
            if self._entity_activity_now(en) < eps:
                to_drop.append(e)
        for e in to_drop:
            for lid in list(self.entity_to_logs.get(e, deque())):
                if lid in self.log_to_entities:
                    self.log_to_entities[lid].discard(e)
            self.entity_to_logs.pop(e, None)
            self.entities.pop(e, None)

    def get_log(self, log_id: int) -> Optional[LogNode]:
        return self.logs.get(log_id)

    def get_entities_for_log(self, log_id: int) -> Set[str]:
        return set(self.log_to_entities.get(log_id, set()))

    def get_logs_for_entity(self, entity: str) -> List[int]:
        dq = self.entity_to_logs.get(entity)
        if not dq:
            return []
        return list(dq)

    def entity_degree(self, entity: str) -> int:
        dq = self.entity_to_logs.get(entity)
        return len(dq) if dq else 0
