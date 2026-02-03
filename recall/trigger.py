from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from .config import RecallConfig
from .recurrence import TemplateBurstDetector
from .text import mask_for_template_key

def severity_level(cfg: RecallConfig, message: str) -> int:
    s = (message or '').lower()
    for kw in cfg.severity_keywords_fatal:
        if kw in s:
            return 3
    for kw in cfg.severity_keywords_error:
        if kw in s:
            return 2
    if 'warn' in s or 'warning' in s:
        return 1
    return 0

@dataclass
class TriggerDecision:
    triggered: bool
    by: str
    template_key: Optional[str] = None

class TriggerEngine:

    def __init__(self, cfg: RecallConfig) -> None:
        self.cfg = cfg
        self.burst = TemplateBurstDetector(burst_window_sec=cfg.burst_window_sec, ema_alpha=cfg.burst_ema_alpha, sigma=cfg.burst_sigma)

    def check(self, ts_sec: int, message: str) -> TriggerDecision:
        msg = message or ''
        if self.cfg.enable_severity_trigger:
            low = msg.lower()
            if any((kw in low for kw in self.cfg.trigger_keywords or [])):
                return TriggerDecision(triggered=True, by='severity', template_key=None)
        if self.cfg.enable_burst_trigger:
            key = mask_for_template_key(msg)
            if self.burst.push_and_check(ts_sec=int(ts_sec), template_key=key):
                return TriggerDecision(triggered=True, by='burst', template_key=key)
            return TriggerDecision(triggered=False, by='none', template_key=key)
        return TriggerDecision(triggered=False, by='none', template_key=None)
