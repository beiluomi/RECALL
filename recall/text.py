from __future__ import annotations
import re
from typing import Iterable, List, Set
_WS = re.compile('\\s+')

def normalize_message_for_dedup(msg: str, case_insensitive: bool) -> str:
    s = (msg or '').strip()
    if case_insensitive:
        s = s.lower()
    s = _WS.sub(' ', s)
    return s

def mask_for_template_key(msg: str) -> str:
    s = (msg or '').strip().lower()
    s = re.sub('\\b0x[0-9a-f]+\\b', '<HEX>', s)
    s = re.sub('\\b\\d+\\b', '<NUM>', s)
    s = _WS.sub(' ', s)
    return s

def tokenize_for_entity_candidates(msg: str) -> List[str]:
    if not msg:
        return []
    toks = _WS.split(msg.strip())
    out: List[str] = []
    for t in toks:
        if not t:
            continue
        while t and (not t[-1].isalnum()) and (t[-1] not in ('/', '.', '_', '-', ':')):
            t = t[:-1]
        while t and (not t[0].isalnum()) and (t[0] not in ('/', '.', '_', '-', ':')):
            t = t[1:]
        if t:
            out.append(t)
    return out

def token_complexity(token: str, case_sensitive: bool=False) -> int:
    if not token:
        return 0
    if len(token) < 2:
        return 0

    def _t(ch: str) -> int:
        if ch.isdigit():
            return 0
        if ch.isalpha():
            return 3 if (case_sensitive and ch.isupper()) else 1
        return 2

    tc = 0
    last_type = _t(token[0])
    for ch in token[1:]:
        cur = _t(ch)
        if cur != last_type:
            tc += 1
        last_type = cur
    return int(tc)

def unique_tokens(tokens: Iterable[str]) -> Set[str]:
    return set((t for t in tokens if t))
