from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@dataclass
class LlmDecision:
    label: str
    confidence: float
    evidence_ids: list
    rationale: str
    raw: str = ''
    error: Optional[str] = None

class DeepSeekClient:

    def __init__(self, api_key: Optional[str]=None, api_key_file: Optional[str]=None, endpoint: str='http://scc.ustc.edu.cn/portal/api/ask', model: str='deepseek-v3', timeout_sec: int=60, max_connections: int=32, temperature: float=0.1, max_tokens: int=256) -> None:
        key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not key and api_key_file:
            try:
                with open(api_key_file, 'r') as f:
                    key = f.read().strip()
            except Exception:
                key = None
        if not key and os.getenv('DEEPSEEK_API_KEY_FILE'):
            try:
                with open(os.getenv('DEEPSEEK_API_KEY_FILE'), 'r') as f:
                    key = f.read().strip()
            except Exception:
                key = None
        if not key:
            raise ValueError('DeepSeek API key required (DEEPSEEK_API_KEY env or --api_key/--api_key_file)')
        self.api_key = key
        self.endpoint = endpoint.rstrip('/')
        self.model = model
        self.timeout_sec = int(timeout_sec)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.session = self._create_session(max_connections=max_connections)

    def _create_session(self, max_connections: int) -> requests.Session:
        s = requests.Session()
        retry_strategy = Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['POST'])
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=max_connections, pool_connections=max_connections, pool_block=False)
        s.mount('http://', adapter)
        s.mount('https://', adapter)
        return s

    def chat(self, prompt: str) -> str:
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload = {'model': self.model, 'stream': False, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': self.temperature, 'max_tokens': self.max_tokens}
        resp = self.session.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_content(data)

    @staticmethod
    def _extract_content(response: Dict[str, Any]) -> str:
        if isinstance(response, dict):
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if isinstance(choice, dict):
                    if 'message' in choice and isinstance(choice['message'], dict):
                        return str(choice['message'].get('content', ''))
                    if 'content' in choice:
                        return str(choice.get('content', ''))
            if 'message' in response and isinstance(response['message'], dict):
                return str(response['message'].get('content', ''))
            if 'content' in response and isinstance(response['content'], str):
                return response['content']
        return ''

def parse_decision(text: str) -> LlmDecision:
    raw = text or ''
    t = raw.strip()
    try:
        if '{' in t and '}' in t:
            js = t[t.index('{'):t.rindex('}') + 1]
            obj = json.loads(js)
            label = str(obj.get('label', 'NORMAL')).strip().upper()
            if label not in ('ANOMALY', 'NORMAL'):
                label = 'ANOMALY' if 'ANOM' in label else 'NORMAL'
            conf = float(obj.get('confidence', 0.5))
            conf = max(0.0, min(1.0, conf))
            eids = obj.get('evidence_ids', []) or []
            if not isinstance(eids, list):
                eids = []
            rationale = str(obj.get('rationale', '') or '')
            return LlmDecision(label=label, confidence=conf, evidence_ids=eids, rationale=rationale, raw=raw)
    except Exception as e:
        return LlmDecision(label='NORMAL', confidence=0.0, evidence_ids=[], rationale='', raw=raw, error=str(e))
    return LlmDecision(label='NORMAL', confidence=0.0, evidence_ids=[], rationale='', raw=raw, error='unparseable')

class LocalHfClient:

    def __init__(self, model_path: str, device: str='auto', dtype: str='auto', max_new_tokens: int=256) -> None:
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f'Local HF model_path not found: {model_path}')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.model_path = str(mp)
        self.max_new_tokens = int(max_new_tokens)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        device_map = 'auto'
        if device and device.lower() in ('cpu', 'cuda'):
            device_map = None
        torch_dtype = None
        if dtype and dtype != 'auto':
            torch_dtype = getattr(torch, dtype, None)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, device_map=device_map, torch_dtype=torch_dtype)
        self.model.eval()
        self.device = device

    def _build_input(self, prompt: str) -> Dict[str, Any]:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{'role': 'user', 'content': prompt}]
            try:
                txt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return {'text': txt}
            except Exception:
                pass
        return {'text': prompt}

    def chat(self, prompt: str) -> str:
        inp = self._build_input(prompt)
        text = inp['text']
        tok = self.tokenizer(text, return_tensors='pt')
        if self.device and self.device.lower() in ('cpu', 'cuda'):
            tok = {k: v.to(self.model.device) for (k, v) in tok.items()}
        else:
            tok = {k: v.to(self.model.device) for (k, v) in tok.items()}
        with self.torch.no_grad():
            out = self.model.generate(**tok, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=getattr(self.tokenizer, 'pad_token_id', None) or getattr(self.tokenizer, 'eos_token_id', None))
        gen = out[0]
        prompt_len = tok['input_ids'].shape[-1]
        suffix = gen[prompt_len:]
        return self.tokenizer.decode(suffix, skip_special_tokens=True).strip()
