"""GPU acceptance check for the paper's four generation models.

Runs each model sequentially, writes real collection artifacts, then checks every
saved token against an independent causal-prefix forward. No benchmark fitting.
    uv run python scripts/check_paper_collection.py --output runs/paper_check
"""
import argparse
import gc
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from openact_collect import CollectionRunner, GenericTask, ModelRunner
from openact_collect.config import load_run_config
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tracing.module_resolver import find_final_norm
from openact_core import Run

MODELS = {
    'llama32': ('meta-llama/Llama-3.2-1B-Instruct', 17, 2048),
    'qwen2': ('Qwen/Qwen2-7B-Instruct', 29, 3584),
    'llama3': ('meta-llama/Meta-Llama-3-8B-Instruct', 33, 4096),
    'llama2': ('meta-llama/Llama-2-7b-chat-hf', 33, 4096),
}


def compare_vectors(actual, expected, tolerance):
    """Use relative vector error; bf16 kernels can vary with sequence length."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f'Invalid activation arrays: {actual.shape}, {expected.shape}')
    error = np.linalg.norm(actual - expected, axis=-1)
    scale = np.maximum(np.linalg.norm(expected, axis=-1), 1e-12)
    relative_error = float(np.max(error / scale))
    if tolerance is not None and relative_error > tolerance:
        raise AssertionError(f'Causal-prefix relative L2 error {relative_error:.6g} > {tolerance}')
    return relative_error


def verify_saved_sample(runner, sample, tolerance=0.03, prefix_lengths=None, fixed_shape=False):
    """Verify all layers, prompt-last, all response positions, means, and RMS sides."""
    prompt = sample.prompt_token_ids.tolist()
    tokens = sample.token_ids.tolist()
    if not prompt or not tokens:
        raise AssertionError('Expected nonempty prompt and generated token sequence')
    n_layers = runner.probed_n_layers
    width = runner.get_model_spec().hidden_dim
    if sample.hidden_states.shape != (len(tokens), n_layers, width):
        raise AssertionError(f'Incomplete token/layer coverage: {sample.hidden_states.shape}')
    # The last prompt token includes the model's assistant-start chat markers.
    rendered_ids = runner.apply_chat_template([{'role': 'user', 'content': sample.prompt_text}])[0].tolist()
    if prompt != rendered_ids:
        raise AssertionError('Saved prompt IDs differ from the selected tokenizer chat template')
    np.testing.assert_allclose(sample.mean_hidden_states, sample.hidden_states.mean(0), rtol=1e-5, atol=1e-6)
    for side in ('pre', 'post'):
        np.testing.assert_allclose(sample.get_final_norm_states(side, 'mean'),
                                   sample.get_final_norm_states(side).mean(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(sample.hidden_states[:, -1], sample.get_final_norm_states('post'))
    np.testing.assert_array_equal(sample.prompt_last_hidden_states[-1], sample.get_final_norm_states('post', 'prompt_last'))

    norm_values = {}
    def capture_norm(module, args, output):
        norm_values['pre'] = args[0][0].detach().float().cpu().numpy().copy()
        norm_values['post'] = output[0].detach().float().cpu().numpy().copy()
    hook = find_final_norm(runner.model).register_forward_hook(capture_norm)
    errors = []
    lengths = list(range(len(tokens) + 1)) if prefix_lengths is None else sorted(set(prefix_lengths))
    if not lengths or any(n < 0 or n > len(tokens) for n in lengths):
        raise ValueError('Prefix lengths must be within the saved response')
    forward_kwargs = {'logits_to_keep': 1} if 'logits_to_keep' in inspect.signature(runner.model.forward).parameters else {}
    fixed_errors = []
    try:
        if fixed_shape:
            # First replay the entire saved sequence, comparing every layer and
            # token. Then change future token IDs WITHOUT changing tensor shape.
            # This isolates causal leakage from bf16 length-dependent GEMM/kernel
            # rounding. Exact equality is required for both checks.
            p = len(prompt)
            for n in [None] + [n for n in lengths if n < len(tokens)]:
                ids = torch.tensor([prompt + tokens], device=runner.device)
                if n is not None:
                    ids[:, p + n:] = (ids[:, p + n:] + 1) % runner.config.vocab_size
                with torch.inference_mode():
                    output = runner.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                          output_hidden_states=True, use_cache=False, **forward_kwargs)
                positions = slice(p - 1, None) if n is None else p + n - 1
                expected = torch.stack([h[0, positions].float().cpu() for h in output.hidden_states],
                                       dim=-2).numpy()
                saved = (np.concatenate([sample.prompt_last_hidden_states[None], sample.hidden_states])
                         if n is None else sample.prompt_last_hidden_states if n == 0
                         else sample.hidden_states[n - 1])
                fixed_errors.append(compare_vectors(saved, expected, 0.0))
                for side in ('pre', 'post'):
                    saved_norm = (np.concatenate([sample.get_final_norm_states(side, 'prompt_last')[None],
                                                  sample.get_final_norm_states(side)])
                                  if n is None else sample.get_final_norm_states(side, 'prompt_last')
                                  if n == 0 else sample.get_final_norm_states(side)[n - 1])
                    fixed_errors.append(compare_vectors(saved_norm, norm_values[side][positions], 0.0))
                del output
        # n=0 is an independent prompt-only forward. n>0 excludes every future
        # generated token. With fixed-shape checks enabled, these different-shape
        # bf16 comparisons are diagnostics, not an arbitrary acceptance threshold.
        for n in lengths:
            ids = torch.tensor([prompt + tokens[:n]], device=runner.device)
            with torch.inference_mode():
                output = runner.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      output_hidden_states=True, use_cache=False, **forward_kwargs)
            expected = torch.stack([h[0, -1].float().cpu() for h in output.hidden_states]).numpy()
            saved = sample.prompt_last_hidden_states if n == 0 else sample.hidden_states[n - 1]
            errors.append(compare_vectors(saved, expected, None if fixed_shape else tolerance))
            for side in ('pre', 'post'):
                saved_norm = (sample.get_final_norm_states(side, 'prompt_last') if n == 0
                              else sample.get_final_norm_states(side)[n - 1])
                errors.append(compare_vectors(saved_norm, norm_values[side][-1], None if fixed_shape else tolerance))
            del output
    finally:
        hook.remove()
    return {'prompt_tokens': len(prompt), 'generated_tokens': len(tokens),
            'hidden_shape': list(sample.hidden_states.shape), 'finish_reason': sample.finish_reason,
            'max_relative_l2_error': max(errors), 'prefix_forwards_checked': len(lengths),
            'prefix_lengths_checked': lengths,
            'fixed_shape_exact_replay_and_causality': fixed_shape,
            'fixed_shape_max_relative_l2_error': max(fixed_errors) if fixed_errors else None,
            'variable_shape_prefix_exceeds_nominal_tolerance': max(errors) > tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', choices=MODELS, help='Repeat to select models; default: all four')
    parser.add_argument('--output', type=Path, default=Path('runs/paper_check'))
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/research.toml'))
    parser.add_argument('--max-tokens', type=int, default=16, help='1..64; every prefix is replayed')
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 64:
        parser.error('--max-tokens must be between 1 and 64 for this acceptance check')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('This check requires CUDA with bf16 support. CPU numerical tests are in pytest.')
    config = load_run_config(args.config)
    capture = CaptureSpec(**config.get('capture', {}))
    if (capture.hidden_states_layers is not None or capture.hidden_states_dtype != 'float32'
            or not all([capture.hidden_states, capture.final_norm, capture.save_per_token,
                        capture.save_mean_states, capture.save_prompt_last])):
        parser.error('Use all layers, float32 storage, both norm sides, per-token, mean, and prompt-last capture')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'relative_l2_tolerance': 0.03, 'models': [], 'passed': False}
    try:
        for alias in args.model or MODELS:
            model_id, layers, width = MODELS[alias]
            row = {'model': model_id, 'passed': False}
            report['models'].append(row)
            runner = ModelRunner(model_id, dtype='bfloat16', device_map='cuda:0', attn_implementation='sdpa')
            print(f'Checking {model_id}', flush=True)
            torch.cuda.reset_peak_memory_stats()
            try:
                runner.load()
                if (runner.probed_n_layers, runner.get_model_spec().hidden_dim) != (layers, width):
                    raise AssertionError('Model dimensions differ from the audited paper checkpoint')
                task = GenericTask([
                    {'prompt': 'What is 17 + 28?', 'answer': '45'},
                    {'prompt': 'Translate “你好，世界” into English.', 'answer': 'Hello, world'},
                ], template='zot', template_name='paper_check_zot', template_text='{prompt}\n\nPlease reason step by step.')
                stats = CollectionRunner(runner, task, args.output / alias, capture_spec=capture,
                                         generation_spec=GenerationSpec(max_new_tokens=args.max_tokens),
                                         run_config=config).run()
                run = Run(args.output / alias)
                if run.validate() or stats['ok'] != 2:
                    raise AssertionError('Collection/storage validation failed')
                row.update(revision=run.manifest.model.revision, samples=[verify_saved_sample(runner, s) for s in run],
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), passed=True)
                print(f'PASS {alias}: every saved position checked against its causal prefix', flush=True)
            except Exception as exc:
                row['error'] = str(exc)
                raise
            finally:
                runner.unload()
                gc.collect()
        report['passed'] = True
    finally:
        (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
