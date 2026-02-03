from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, Iterable, Any
from .config import RecallConfig
from .recurrence import TokenRecurrenceCounter
from .text import tokenize_for_entity_candidates, token_complexity
_IPV4_PORT = re.compile('^(?P<ip>(?:\\d{1,3}\\.){3}\\d{1,3})(?::\\d{1,5})?$')

def classify_entity_type(ent: str) -> str:
    s = (ent or '').strip()
    if not s:
        return 'unknown'
    if _IPV4_PORT.match(s):
        return 'ip'
    if s.startswith('/') or s.startswith('./'):
        return 'path'
    if s.startswith('blk_'):
        return 'block_id'
    if re.match('^[A-Za-z]\\w*-\\w+', s):
        return 'identifier'
    if re.match('^\\d+$', s):
        return 'number'
    if re.match('^[A-Z0-9_]{3,}$', s):
        return 'code'
    return 'token'

@dataclass
class EntityExtractionResult:
    estat: Set[str]
    esem: Set[str]
    final: Set[str]
    estat_validated: Set[str] = field(default_factory=set)

@dataclass
class SemanticValidationResult:
    keep: Set[str]
    add: Set[str]
    drop: Set[str]

def _extract_values(items: Any) -> Set[str]:
    out: Set[str] = set()
    if not items:
        return out
    if isinstance(items, list):
        for it in items:
            if isinstance(it, str):
                v = it.strip()
                if v:
                    out.add(v)
            elif isinstance(it, dict):
                v = (it.get('value') or it.get('entity') or it.get('text') or '').strip()
                if v:
                    out.add(v)
    elif isinstance(items, dict):
        v = (items.get('value') or items.get('entity') or items.get('text') or '').strip()
        if v:
            out.add(v)
    return out

def _in_message_or_ip_port(entity: str, message: str) -> bool:
    e = (entity or '').strip()
    if not e:
        return False
    msg = message or ''
    if e in msg:
        return True
    m = _IPV4_PORT.match(e)
    if m:
        ip = m.group('ip')
        if ip and (ip in msg):
            return True
        if ip and (f'{ip}:' in msg):
            return True
    return False

def parse_semantic_validation_response(text: str, message: str) -> SemanticValidationResult:
    raw = (text or '').strip()
    keep: Set[str] = set()
    add: Set[str] = set()
    drop: Set[str] = set()
    try:
        if '{' not in raw or '}' not in raw:
            return SemanticValidationResult(keep=set(), add=set(), drop=set())
        js = raw[raw.index('{'):raw.rindex('}') + 1]
        obj = json.loads(js)
        if isinstance(obj, dict):
            if any((k in obj for k in ('keep', 'add', 'drop'))):
                keep = _extract_values(obj.get('keep'))
                add = _extract_values(obj.get('add'))
                drop = _extract_values(obj.get('drop'))
            elif 'entities' in obj:
                ents = obj.get('entities')
                vals = _extract_values(ents)
                add = set(vals)
            elif 'final' in obj:
                vals = _extract_values(obj.get('final'))
                keep = set(vals)
        keep = {e for e in keep if _in_message_or_ip_port(e, message)}
        add = {e for e in add if _in_message_or_ip_port(e, message)}
        drop = {e for e in drop if e}
        return SemanticValidationResult(keep=keep, add=add, drop=drop)
    except Exception:
        return SemanticValidationResult(keep=set(), add=set(), drop=set())

class StatisticalEntityExtractor:

    def __init__(self, cfg: RecallConfig) -> None:
        self.cfg = cfg
        self.rf_counter = TokenRecurrenceCounter(window_sec=cfg.delta_t_sec)

    def extract(self, ts_sec: int, message: str) -> Set[str]:
        toks = tokenize_for_entity_candidates(message)
        self.rf_counter.push(int(ts_sec), toks)
        ents: Set[str] = set()
        for tok in toks:
            if not tok or len(tok) < self.cfg.min_token_len:
                continue
            if self.cfg.should_drop_token(tok):
                continue
            m = _IPV4_PORT.match(tok)
            if m:
                ip_only = m.group('ip')
                if ip_only and (not self.cfg.is_blacklisted_entity(ip_only)):
                    ents.add(ip_only)
            if self.cfg.is_blacklisted_entity(tok):
                continue
            tc = token_complexity(tok, case_sensitive=False)
            rf = self.rf_counter.rf(tok)
            if tc > self.cfg.theta_tc and rf > self.cfg.theta_rf:
                ents.add(tok)
        return ents

class SemanticEntityExtractor:

    def __init__(self, cfg: RecallConfig, backend: str, api_key: Optional[str]=None, api_key_file: Optional[str]=None, local_model_path: Optional[str]=None) -> None:
        from .llm_client import DeepSeekClient, LocalHfClient
        self.cfg = cfg
        b = (backend or '').strip().lower()
        if b in ('deepseek', 'deepseek_api', 'api'):
            self.client = DeepSeekClient(api_key=api_key, api_key_file=api_key_file, endpoint=cfg.llm_endpoint, model=cfg.llm_model_name)
        elif b in ('local', 'local_hf', 'hf'):
            if not local_model_path:
                raise ValueError('local semantic backend requires --semantic_model_path')
            self.client = LocalHfClient(model_path=local_model_path, max_new_tokens=256)
        else:
            raise ValueError(f'Unknown semantic backend: {backend}')

    def validate_and_supplement(self, message: str, candidates: Iterable[str]) -> SemanticValidationResult:
        cand_list = [c.strip() for c in (candidates or []) if c and str(c).strip()]
        prompt = (
            '[Task]\n'
            'Validate and supplement key system entities for log diagnosis.\n'
            'You will be given a log message and a list of candidate entities produced by a statistical filter.\n'
            'Do two things:\n'
            '1) KEEP: candidates that are real key entities mentioned in the log and useful for linking diagnostically related logs.\n'
            '2) DROP: candidates that are not entities / too generic / artifacts.\n'
            '3) ADD: entities mentioned in the log but missing from the candidates.\n'
            '\n'
            'Entity types include: IP address, hostname/node ID, process ID, file path, request/transaction ID, error code.\n'
            '\n'
            '[Log Message]\n'
            f'{message}\n'
            '\n'
            '[Candidate Entities]\n'
            f'{json.dumps(cand_list, ensure_ascii=False)}\n'
            '\n'
            '[Output Requirements]\n'
            'Output ONLY valid JSON in this schema:\n'
            '{\n'
            '  "keep": [{"value": "...", "type": "..."}],\n'
            '  "drop": [{"value": "...", "reason": "..."}],\n'
            '  "add":  [{"value": "...", "type": "..."}]\n'
            '}\n'
            'Rules:\n'
            '- Prefer values that appear verbatim in the log message.\n'
            '- No duplicates.\n'
        )
        out = self.client.chat(prompt)
        return parse_semantic_validation_response(out, message=message)

    def extract(self, message: str) -> Set[str]:
        res = self.validate_and_supplement(message=message, candidates=[])
        return set(res.keep) | set(res.add)

def extract_entities(cfg: RecallConfig, stat_extractor: StatisticalEntityExtractor, ts_sec: int, message: str, semantic_extractor: Optional[SemanticEntityExtractor]=None) -> EntityExtractionResult:
    estat = stat_extractor.extract(ts_sec, message)
    estat_validated = set(estat)
    esem: Set[str] = set()
    if cfg.enable_semantic_channel and semantic_extractor is not None:
        if len(estat) < cfg.semantic_trigger_min_entities:
            sem = semantic_extractor.validate_and_supplement(message=message, candidates=estat)
            if sem.keep:
                estat_validated = set(sem.keep)
            esem = set(sem.add)
    final = set(estat_validated) | set(esem)
    final = {e for e in final if not cfg.is_blacklisted_entity(e)}
    estat_validated = {e for e in estat_validated if not cfg.is_blacklisted_entity(e)}
    return EntityExtractionResult(estat=estat, estat_validated=estat_validated, esem=esem, final=final)
