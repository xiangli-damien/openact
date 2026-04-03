from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from openact_core.io.sample import Sample
from openact_eval.evaluators.base import Evaluator, EvalRecord, extract_sample_context_meta

_DEFAULT_JUDGE_PROMPT = 'You are an expert evaluator of model outputs.\n\n## Task\nGiven a question, a reference (ground-truth) answer, and a model\'s response, determine whether the model\'s response is **correct**.\n\n- Focus on the **substance** of the answer, not surface formatting.\n- Numerical answers that match within reasonable tolerance are correct.\n- Equivalent mathematical expressions are correct.\n\n## Input\n**Question:**\n{prompt}\n\n**Reference answer:**\n{ground_truth}\n\n**Model response:**\n{response}\n\n## Output\nRespond with a JSON object and nothing else:\n{{"is_correct": true, "reasoning": "brief explanation"}}\nor\n{{"is_correct": false, "reasoning": "brief explanation"}}\n'
_ENV_KEY_MAP = {'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}


def _resolve_api_key(backend: str, explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    env_var = _ENV_KEY_MAP.get(backend)
    if env_var:
        value = os.environ.get(env_var)
        if value:
            return value
    config_path = Path.home() / '.openact' / 'keys.json'
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            return data.get(backend) or data.get(env_var)
        except (json.JSONDecodeError, KeyError):
            pass
    return None


class LLMJudgeEvaluator(Evaluator):
    def __init__(self, backend: str = 'openai', model: str = 'gpt-4o-mini', api_key: Optional[str] = None, judge_prompt: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 512, max_retries: int = 3, retry_delay: float = 1.0, parse_fn: Optional[Callable[[str], Tuple[bool, str]]] = None):
        self.backend = backend.lower()
        self.model = model
        self._explicit_key = api_key
        self.judge_prompt = judge_prompt or _DEFAULT_JUDGE_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.parse_fn = parse_fn or self._default_parse
        self._client = None

    @property
    def name(self) -> str:
        return f'llm_judge/{self.backend}/{self.model}'

    def setup(self) -> None:
        api_key = _resolve_api_key(self.backend, self._explicit_key)
        if self.backend == 'openai':
            try:
                import openai
            except ImportError as exc:
                raise ImportError('openai package required. Install with: pip install openact-eval[llm]') from exc
            if not api_key:
                raise ValueError('OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key= to the evaluator.')
            self._client = openai.OpenAI(api_key=api_key)
        elif self.backend == 'anthropic':
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError('anthropic package required. Install with: pip install openact-eval[llm]') from exc
            if not api_key:
                raise ValueError('Anthropic API key not found. Set ANTHROPIC_API_KEY env var or pass api_key= to the evaluator.')
            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Supported: openai, anthropic")

    def teardown(self) -> None:
        self._client = None

    def evaluate_sample(self, sample: Sample) -> EvalRecord:
        prompt_text = sample.prompt_text or ''
        response_text = sample.response_text or ''
        ground_truth = sample.ground_truth or ''
        judge_input = self.judge_prompt.format(prompt=prompt_text, ground_truth=ground_truth, response=response_text)
        judge_output = self._call_api(judge_input)
        try:
            is_correct, reasoning = self.parse_fn(judge_output)
        except Exception as exc:
            meta = extract_sample_context_meta(sample)
            return EvalRecord(sample_idx=sample.sample_idx, ground_truth=ground_truth, error=f'Parse error: {exc}. Raw: {judge_output[:200]}', meta=meta)
        meta = extract_sample_context_meta(sample)
        return EvalRecord(sample_idx=sample.sample_idx, is_correct=is_correct, extracted_answer=None, ground_truth=ground_truth, score=1.0 if is_correct else 0.0, judge_reasoning=reasoning, meta=meta)

    def _call_api(self, content: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.backend == 'openai':
                    return self._call_openai(content)
                if self.backend == 'anthropic':
                    return self._call_anthropic(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f'API call failed after {self.max_retries} retries: {last_error}')

    def _call_openai(self, content: str) -> str:
        response = self._client.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': content}], temperature=self.temperature, max_tokens=self.max_tokens)
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, content: str) -> str:
        response = self._client.messages.create(model=self.model, messages=[{'role': 'user', 'content': content}], temperature=self.temperature, max_tokens=self.max_tokens)
        return response.content[0].text.strip()

    @staticmethod
    def _default_parse(output: str) -> Tuple[bool, str]:
        candidates = [output]
        candidates.extend(re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                return bool(data['is_correct']), str(data.get('reasoning', ''))
            except (json.JSONDecodeError, KeyError):
                continue
        raise ValueError(f'Cannot parse judge output as JSON: {output[:200]}')

    def _get_config(self) -> Dict[str, Any]:
        return {'evaluator_class': self.__class__.__name__, 'backend': self.backend, 'model': self.model, 'temperature': self.temperature, 'max_tokens': self.max_tokens}
