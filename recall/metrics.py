from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        (p, r) = (self.precision, self.recall)
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> Dict:
        return {'tp': self.tp, 'fp': self.fp, 'tn': self.tn, 'fn': self.fn, 'precision': self.precision, 'recall': self.recall, 'f1': self.f1, 'total': self.tp + self.fp + self.tn + self.fn}

def compute_metrics(true_labels: List[int], pred_labels: List[int]) -> Metrics:
    tp = fp = tn = fn = 0
    for (t, p) in zip(true_labels, pred_labels):
        if t == 1:
            if p == 1:
                tp += 1
            else:
                fn += 1
        elif p == 1:
            fp += 1
        else:
            tn += 1
    return Metrics(tp=tp, fp=fp, tn=tn, fn=fn)
