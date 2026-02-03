from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Set, Tuple
import math
from .text import unique_tokens

class TokenRecurrenceCounter:

    def __init__(self, window_sec: int) -> None:
        self.window_sec = int(window_sec)
        self._q: Deque[Tuple[int, Set[str]]] = deque()
        self._counts: Dict[str, int] = defaultdict(int)

    def push(self, ts_sec: int, tokens_in_log: Iterable[str]) -> None:
        ts = int(ts_sec)
        tok_set = unique_tokens(tokens_in_log)
        self._q.append((ts, tok_set))
        for t in tok_set:
            self._counts[t] += 1
        self._evict(ts)

    def _evict(self, now_ts: int) -> None:
        if self.window_sec <= 0:
            return
        cutoff = int(now_ts) - self.window_sec
        while self._q and self._q[0][0] < cutoff:
            (_, tok_set) = self._q.popleft()
            for t in tok_set:
                self._counts[t] -= 1
                if self._counts[t] <= 0:
                    self._counts.pop(t, None)

    def rf(self, token: str) -> int:
        return int(self._counts.get(token, 0))

@dataclass
class _EmaStats:
    mean: float = 0.0
    var: float = 0.0
    initialized: bool = False

class TemplateBurstDetector:

    def __init__(self, burst_window_sec: int, ema_alpha: float, sigma: float=3.0) -> None:
        self.burst_window_sec = int(burst_window_sec)
        self.ema_alpha = float(ema_alpha)
        self.sigma = float(sigma)
        self._q: Deque[Tuple[int, str]] = deque()
        self._win_counts: Dict[str, int] = defaultdict(int)
        self._ema: Dict[str, _EmaStats] = defaultdict(_EmaStats)

    def push_and_check(self, ts_sec: int, template_key: str) -> bool:
        ts = int(ts_sec)
        key = template_key or ''
        if not key:
            return False
        self._q.append((ts, key))
        self._win_counts[key] += 1
        self._evict(ts)
        x = float(self._win_counts.get(key, 0))
        st = self._ema[key]
        a = self.ema_alpha
        if not st.initialized:
            st.mean = x
            st.var = 0.0
            st.initialized = True
        else:
            prev_mean = st.mean
            st.mean = (1.0 - a) * st.mean + a * x
            st.var = (1.0 - a) * st.var + a * (x - prev_mean) * (x - st.mean)
        std = math.sqrt(max(st.var, 0.0))
        threshold = st.mean + self.sigma * std
        if threshold <= 1.0:
            return False
        return x > threshold

    def _evict(self, now_ts: int) -> None:
        if self.burst_window_sec <= 0:
            return
        cutoff = int(now_ts) - self.burst_window_sec
        while self._q and self._q[0][0] < cutoff:
            (_, key) = self._q.popleft()
            self._win_counts[key] -= 1
            if self._win_counts[key] <= 0:
                self._win_counts.pop(key, None)
