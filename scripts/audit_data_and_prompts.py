"""CPU preflight: scan benchmark rows, render every variant, test target tokenizers.

Run from the repository root:
    uv run python scripts/audit_data_and_prompts.py --output reports/data_and_prompts.json
Downloads datasets/tokenizers, never model weights. Reports failures with a nonzero exit.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from openact_collect.engine.model_manager import ModelRunner
from openact_collect.engine.offset_calculator import OffsetCalculator
from openact_collect.schema import GenerationProfile
from openact_collect.tasks.capability.belebele import BELEBELE_LANG_MAP
from openact_collect.tasks.capability.mgsm import MGSM_LANGUAGES
from openact_collect.tasks.registry import TaskRegistry
from openact_core.tasks.templates import get_templates

MODELS = [
    'meta-llama/Llama-3.2-1B-Instruct', 'Qwen/Qwen2-7B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-7b-chat-hf',
]
MONO = ['gsm8k', 'mmlu', 'math', 'theoremqa', 'arc_challenge', 'commonsenseqa',
        'truthfulqa', 'humaneval', 'ifeval']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/data_and_prompts.json')
    args = parser.parse_args()
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'models': [], 'datasets': [], 'errors': []}
    runners = []
    for model_id in MODELS:
        try:
            runner = ModelRunner(model_id)
            config = AutoConfig.from_pretrained(model_id)
            revision = getattr(config, '_commit_hash', None)
            runner._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            runner._loaded = True
            generation = GenerationConfig.from_pretrained(model_id, revision=revision)
            row = {'id': model_id, 'revision': getattr(config, '_commit_hash', None),
                   'decoder_layers': config.num_hidden_layers,
                   'hidden_state_entries': config.num_hidden_layers + 1,
                   'hidden_dim': config.hidden_size,
                   'eos_token_id': generation.eos_token_id,
                   'template_sha256': hashlib.sha256(str(runner.tokenizer.chat_template).encode()).hexdigest(),
                   'prompt_checks': 0, 'offset_checks': 0}
            for text in ['hello hello hello', '你好，世界！', 'مرحبا بالعالم', 'বাংলা ভাষা', 'hello 👩🏽‍💻 world']:
                ids = runner.encode(text)
                decoded = runner.decode(ids)
                calculator = OffsetCalculator(runner.tokenizer)
                offsets = calculator.compute_offsets(decoded, ids)
                assert calculator.validate_offsets(decoded, ids, offsets)[0]
                # Force the fallback as generated tokens need not match re-encoding.
                calculator._has_fast_tokenizer = False
                offsets = calculator.compute_offsets(decoded, ids)
                assert calculator.validate_offsets(decoded, ids, offsets)[0]
                row['offset_checks'] += 2
            runners.append((runner, row))
            report['models'].append(row)
            print(f'Tokenizer OK: {model_id}', flush=True)
        except Exception as exc:
            report['errors'].append({'model': model_id, 'error': str(exc)})
    configs = [(task, {}) for task in MONO]
    configs += [('mgsm', {'language': lang}) for lang in MGSM_LANGUAGES]
    configs += [('belebele', {'language': lang}) for lang in BELEBELE_LANG_MAP]
    configs += [(task, {'profiles': {'greedy': GenerationProfile(name='greedy')}, **({'include_artifacts': False} if task == 'jbb' else {})}) for task in ['jbb', 'advbench', 'xstest']]
    for name, kwargs in configs:
        label = name + (':' + kwargs['language'] if 'language' in kwargs else '')
        try:
            task = TaskRegistry.create(name, **kwargs)
            counts = Counter()
            n_rows = 0
            for index, item in enumerate(task.iter_items()):
                assert item.sample_idx == index, f'Noncontiguous sample index {item.sample_idx}'
                assert task.render_prompt(item).strip(), 'Empty prompt'
                if name != 'ifeval':
                    assert item.ground_truth is not None and str(item.ground_truth).strip(), 'Missing label'
                if name in ('mmlu', 'belebele', 'arc_challenge', 'commonsenseqa'):
                    assert item.ground_truth in 'ABCDE', 'Invalid multiple-choice label'
                counts[getattr(item, 'split', None) or item.meta.get('subject') or item.meta.get('category') or 'all'] += 1
                n_rows += 1
            assert n_rows > 0, 'Empty dataset'
            variants = list(get_templates(name)) if name in MONO + ['mgsm', 'belebele'] else ['raw', 'zot']
            for variant in variants:
                task._template_variant = variant
                for item in islice(task.iter_items(), 3):
                    prompt = task.render_prompt(item)
                    assert prompt.strip()
                    for runner, row in runners:
                        messages = [{'role': 'user', 'content': prompt}]
                        tokenized = runner.apply_chat_template(messages)
                        rendered = runner.render_chat_text(messages)
                        expected = runner.tokenizer(rendered, add_special_tokens=False, return_tensors='pt').input_ids
                        assert tokenized.shape[0] == 1 and tokenized.shape[-1] > 0
                        assert tokenized.equal(expected), 'Chat rendering/tokenization mismatch'
                        row['prompt_checks'] += 1
            report['datasets'].append({'task': label, 'rows': n_rows, 'groups': dict(counts),
                                       'variants': variants, 'sources': task.dataset_sources})
            print(f'Dataset OK: {label}: {n_rows} rows, {len(variants)} variants', flush=True)
        except Exception as exc:
            report['errors'].append({'task': label, 'error': str(exc)})
            print(f'FAILED: {label}: {exc}', flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Report: {output}; {len(report["errors"])} failures', flush=True)
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
