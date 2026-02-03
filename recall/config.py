from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Pattern
import re

@dataclass(frozen=True)
class RecallConfig:
    theta_tc: int = 2
    theta_rf: int = 2
    delta_t_sec: int = 300
    min_token_len: int = 2
    graph_window_t_sec: int = 900
    decay_lambda: float | None = None
    theta_w: float = 0.05
    activity_beta: float = 0.99
    activity_alpha: float = 1.0
    activity_epsilon: float = 0.1
    temporal_k: int = 15
    evidence_budget_nmax: int = 30
    degree_threshold_dmax: int = 200
    score_a: float = 1.0
    score_b: float = 1.0
    score_c: float = 1.0
    dedup_case_insensitive: bool = False
    enable_severity_trigger: bool = True
    enable_burst_trigger: bool = True
    burst_sigma: float = 3.0
    burst_window_sec: int = 300
    burst_ema_alpha: float = 0.01
    enable_semantic_channel: bool = False
    semantic_backend: str = 'deepseek_api'
    semantic_local_model_path: str | None = None
    semantic_trigger_min_entities: int = 1
    llm_model_name: str = 'deepseek-v3'
    llm_endpoint: str = '..'
    llm_backend: str = 'deepseek_api'
    llm_local_model_path: str | None = None
    trigger_keywords: List[str] = field(default_factory=lambda : ['fatal', 'panic', 'exception', 'critical', 'failure', 'machine check'])
    severity_keywords_fatal: List[str] = field(default_factory=lambda : ['fatal', 'panic', 'critical', 'machine check'])
    severity_keywords_error: List[str] = field(default_factory=lambda : ['error', 'exception', 'fail', 'failure', 'crash', 'abort', 'terminated'])
    entity_blacklist_exact: List[str] = field(default_factory=lambda : ['127.0.0.1', '0.0.0.0', 'localhost', '/tmp'])
    entity_blacklist_regex: List[Pattern[str]] = field(default_factory=lambda : [re.compile('^::1$')])
    token_drop_regex: List[Pattern[str]] = field(default_factory=lambda : [re.compile('^\\\\d{4}-\\\\d{2}-\\\\d{2}-\\\\d{2}\\\\.\\\\d{2}\\\\.\\\\d{2}\\\\.\\\\d+$')])

    def is_blacklisted_entity(self, ent: str) -> bool:
        s = (ent or '').strip()
        if not s:
            return True
        if s in self.entity_blacklist_exact:
            return True
        for rx in self.entity_blacklist_regex:
            if rx.search(s):
                return True
        return False

    def should_drop_token(self, token: str) -> bool:
        if not token:
            return True
        for rx in self.token_drop_regex:
            if rx.search(token):
                return True
        return False
