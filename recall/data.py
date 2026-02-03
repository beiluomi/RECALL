from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

@dataclass
class LogRecord:
    log_id: int
    ts_sec: int
    message: str
    true_label: int

def _read_labeled_loghub_file(path: Path) -> List[LogRecord]:
    out: List[LogRecord] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            pos = s.find(' ')
            if pos < 0:
                continue
            raw_label = s[:pos]
            rest = s[pos + 1:]
            parts = s.split(' ')
            try:
                ts = int(parts[1])
            except Exception:
                ts = 0
            true = 0 if raw_label == '-' else 1
            out.append(LogRecord(log_id=-1, ts_sec=ts, message=rest, true_label=true))
    out.sort(key=lambda r: r.ts_sec)
    for (i, r) in enumerate(out):
        r.log_id = i
    return out

def load_dataset(dataset: str, loghub_root: Optional[str]=None) -> List[LogRecord]:
    ds = (dataset or '').strip().lower()
    default_root = Path(__file__).resolve().parent.parent.parent / 'LogHub'
    root = Path(loghub_root).resolve() if loghub_root else default_root
    fname_map = {'bgl': 'BGL.log', 'thunderbird': 'TDB.log', 'tdb': 'TDB.log', 'spirit': 'spirit.log'}
    if ds not in fname_map:
        raise ValueError(f'Unsupported dataset: {dataset} (supported: {sorted(fname_map)})')
    path = root / fname_map[ds]
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {path}')
    return _read_labeled_loghub_file(path)
