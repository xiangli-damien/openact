"""Screen natural responses, label with Guard, capture exact selected sequences.

No reference answers, attack search, or synthetic class labels are used. Outputs
are ordinary OpenAct Runs. The balanced subset is NOT a safety-rate estimate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
from transformers import LogitsProcessor

from openact_collect import CollectionRunner, ModelRunner
from openact_collect.config import load_run_config, tomllib
from openact_collect.engine.collector import CollectResult
from openact_collect.engine.model_manager import GenerationResult
from openact_collect.extractors.hidden_state_data import GenerationMetrics, HiddenStateData
from openact_collect.schema import CaptureSpec, GenerationSpec
from openact_collect.tasks.prepared import PreparedParquetTask
from openact_collect.tracing import ActivationRecorder
from openact_collect.tracing.module_resolver import find_final_norm
from openact_core import Run
from openact_core.schema.status import SampleStatus
from openact_eval.evaluators.base import EvalRecord, EvalResult
from openact_eval.evaluators.safety_evaluators import LlamaGuardEvaluator

try:
    from scripts.run_collection_matrix import SelectedPreparedTask
    from scripts.run_math_collection import atomic_json, now, sha256, validate_shard, publish_with_retries, verified_receipt
except ModuleNotFoundError:
    from run_collection_matrix import SelectedPreparedTask
    from run_math_collection import atomic_json, now, sha256, validate_shard, publish_with_retries, verified_receipt


class BatchMetrics(LogitsProcessor):
    """Greedy metrics through the first EOS, including that EOS, per sequence."""
    def __init__(self, eos):
        self.eos = eos
        self.count = self.sums = self.alive = None

    def __call__(self, input_ids, scores):
        if self.count is None:
            self.count = torch.zeros(len(scores), device=scores.device)
            self.sums = torch.zeros((len(scores), 3), device=scores.device)
            self.alive = torch.ones(len(scores), device=scores.device, dtype=torch.bool)
        else:
            for token in self.eos:
                self.alive &= input_ids[:, -1] != token
        lp = scores.float().log_softmax(-1)
        p = lp.exp()
        entropy = -(p * torch.where(torch.isfinite(lp), lp, 0.)).sum(-1)
        maximum = lp.max(-1).values
        values = torch.stack([maximum.exp(), entropy, maximum], -1)
        self.sums += values * self.alive[:, None]
        self.count += self.alive
        return scores

    def result(self):
        values = (self.sums / self.count.clamp_min(1)[:, None]).cpu().tolist()
        return [dict(max_probability=v[0], entropy=v[1], perplexity=math.exp(min(-v[2], 100))) for v in values]


def trim_response(tokens, eos):
    tokens = list(tokens)
    for i, token in enumerate(tokens):
        if token in eos:
            return tokens[:i + 1]
    return tokens


def quota_selection(rows, counts, target):
    counts = dict(counts)
    selected = []
    for row in rows:
        label = row['judge'].get('is_safe')
        if label is None:
            continue
        key = 'safe' if label else 'unsafe'
        if counts[key] < target:
            selected.append(row)
            counts[key] += 1
    return selected, counts


def capture_saved(runner, prompt_ids, token_ids, capture, finish_reason, budget):
    """One teacher-forced forward of stored IDs; never regenerate or retokenize."""
    if not prompt_ids or not token_ids:
        raise ValueError('Both prompt and response IDs are required')
    sequence = torch.tensor([prompt_ids + token_ids], device=runner.device)
    p, t = len(prompt_ids), len(token_ids)
    kwargs = {'logits_to_keep': 1} if 'logits_to_keep' in inspect.signature(runner.model.forward).parameters else {}
    positions = list(range(p - 1, p + t))
    with torch.inference_mode(), ActivationRecorder(runner.model, capture, decoder_n_layers=runner.decoder_n_layers, token_positions=positions) as recorder:
        measured = runner.model(input_ids=sequence, attention_mask=torch.ones_like(sequence),
                                use_cache=False, return_dict=True, output_hidden_states=True, **kwargs)
        layers = tuple(h[:, p - 1:, :].detach().to('cpu', copy=True) for h in measured.hidden_states)
        hidden = [tuple(h[:, i:i + 1, :] for h in layers) for i in range(t + 1)]
        traces = recorder.trace()
        del measured
    return GenerationResult(sequence[:, :p], sequence[:, p:], sequence, p, t, hidden,
                            None, finish_reason, list(token_ids), list(range(1, t + 1)), traces, budget)


def capture_spec_for_mode(base_capture, mode):
    values = dict(base_capture)
    if mode == 'prompt-last':
        values.update(hidden_states=True, final_norm=True, save_prompt_last=True,
                      save_per_token=False, save_mean_states=False, attention=False, mlp=False)
    elif mode != 'full':
        raise ValueError(f'Unknown capture mode: {mode}')
    return CaptureSpec(**values)


def capture_prompt_last(runner, prompt_ids, capture):
    """Forward only the exact prompt; response text is used exclusively for labels."""
    if not prompt_ids:
        raise ValueError('Exact prompt IDs are required')
    sequence = torch.tensor([prompt_ids], device=runner.device)
    kwargs = {'logits_to_keep': 1} if 'logits_to_keep' in inspect.signature(runner.model.forward).parameters else {}
    with torch.inference_mode(), ActivationRecorder(
            runner.model, capture, decoder_n_layers=runner.decoder_n_layers,
            token_positions=[len(prompt_ids) - 1]) as recorder:
        measured = runner.model(input_ids=sequence, attention_mask=torch.ones_like(sequence),
                                use_cache=False, return_dict=True, output_hidden_states=True, **kwargs)
        if len(measured.hidden_states) != runner.probed_n_layers:
            raise ValueError('Unexpected prompt layer count')
        layers = capture.get_effective_layers(runner.probed_n_layers)
        values = torch.stack([measured.hidden_states[i][0, -1] for i in layers]).float().cpu().numpy()
        trace = recorder.trace()
        if len(trace.final_norm_pre) != 1 or len(trace.final_norm_post) != 1:
            raise ValueError('Expected exactly one prompt-last final norm trace')
        return HiddenStateData(prompt_last_states=values,
            final_norm_pre_prompt_last=trace.final_norm_pre[0].float().numpy(),
            final_norm_post_prompt_last=trace.final_norm_post[0].float().numpy(), n_tokens=0)


def validate_prompt_shard(runner, run, ids):
    """Check all rows, then independently replay prompt-only states for two rows."""
    if [s.sample_id for s in run] != ids or len(set(ids)) != len(ids):
        raise ValueError('Prompt shard sample IDs mismatch')
    issues = run.validate()
    if issues or not run.is_complete or run.n_valid != len(ids):
        raise ValueError(f'Invalid prompt shard: {issues}')
    for group in ('hidden_states', 'final_norm/pre', 'final_norm/post'):
        if any(f'{group}/{reduction}' in run._zarr for reduction in ('mean', 'per_token')):
            raise ValueError('Prompt-only shard unexpectedly contains generation activations')
    layers = CaptureSpec(**run.manifest.capture_config).get_effective_layers(runner.probed_n_layers)
    width = runner.get_model_spec().hidden_dim
    for sample in run:
        h = sample.prompt_last_hidden_states
        pre = sample.get_final_norm_states('pre', 'prompt_last')
        post = sample.get_final_norm_states('post', 'prompt_last')
        if h.shape != (len(layers), width) or pre.shape != (width,) or post.shape != (width,):
            raise ValueError('Prompt activation shape mismatch')
        if not all(np.isfinite(v).all() for v in (h, pre, post)):
            raise ValueError('Nonfinite prompt activations')
        if runner.probed_n_layers - 1 in layers:
            np.testing.assert_array_equal(h[layers.index(runner.probed_n_layers - 1)], post)
    checked = sorted(set([0, len(run) - 1]))
    for index in checked:
        sample = run[index]
        ids_tensor = torch.tensor([sample.prompt_token_ids.tolist()], device=runner.device)
        norm = {}
        def save_norm(module, inputs, output):
            norm['pre'] = inputs[0][0, -1].detach().float().cpu().numpy().copy()
            norm['post'] = output[0, -1].detach().float().cpu().numpy().copy()
        hook = find_final_norm(runner.model).register_forward_hook(save_norm)
        try:
            kwargs = {'logits_to_keep': 1} if 'logits_to_keep' in inspect.signature(runner.model.forward).parameters else {}
            with torch.inference_mode():
                reference = runner.model(input_ids=ids_tensor, attention_mask=torch.ones_like(ids_tensor),
                    use_cache=False, output_hidden_states=True, return_dict=True, **kwargs)
                expected = torch.stack([reference.hidden_states[i][0, -1] for i in layers]).float().cpu().numpy()
            np.testing.assert_array_equal(sample.prompt_last_hidden_states, expected)
            for side in ('pre', 'post'):
                np.testing.assert_array_equal(sample.get_final_norm_states(side, 'prompt_last'), norm[side])
            del reference
        finally:
            hook.remove()
    return dict(capture_mode='prompt-last', samples_checked=len(run), samples_deep_checked=checked,
                prompt_only_exact_replay=True, response_hidden_states_stored=False)


class ReplayCollection(CollectionRunner):
    def __init__(self, *args, responses, progress_callback=None, capture_mode='full', **kwargs):
        super().__init__(*args, **kwargs)
        self.responses = responses
        self.progress_callback = progress_callback
        self.capture_mode = capture_mode
        if capture_mode == 'prompt-last' and (self.capture_spec.save_per_token or self.capture_spec.save_mean_states):
            raise ValueError('Prompt-only collection requires prompt-only capture flags')

    def _create_manifest(self):
        manifest = super()._create_manifest()
        manifest.custom['safety_capture_mode'] = self.capture_mode
        if self.capture_mode == 'prompt-last':
            manifest.custom.update(activation_forward_input='prompt_ids_only',
                                   token_alignment='prompt_last', mean_token_policy=None)
        return manifest

    def _process_sample(self, item, prompt_text=None):
        started = time.perf_counter()
        row = self.responses[item.sample_id]
        rendered = self.task.render_prompt(item)
        ids = self.model_manager.apply_chat_template([{'role': 'user', 'content': rendered}])[0].tolist()
        if ids != row['prompt_token_ids'] or rendered != row['prompt_text']:
            raise ValueError('Screening/replay prompt mismatch')
        if self.model_manager.decode(row['token_ids']) != row['response_text']:
            raise ValueError('Screening/replay response mismatch')
        gen = None
        try:
            if self.capture_mode == 'prompt-last':
                hidden = capture_prompt_last(self.model_manager, ids, self.capture_spec)
            else:
                gen = capture_saved(self.model_manager, ids, row['token_ids'], self.capture_spec,
                                    row['finish_reason'], row['effective_max_new_tokens'])
                hidden = self.extractor.extract(gen, input_ids=gen.input_ids)
            result = CollectResult(
                sample_idx=item.sample_idx, status=SampleStatus.OK, response_text=row['response_text'],
                token_ids=row['token_ids'], token_offsets=self.offset_calculator.compute_offsets(row['response_text'], row['token_ids']),
                finish_reason=row['finish_reason'], hidden_state_data=hidden,
                generation_metrics=GenerationMetrics(**row['generation_metrics']),
                meta={**item.meta, 'sample_id': item.sample_id, 'prompt_text': rendered,
                      'model_input_text': row['model_input_text'], 'ground_truth': item.ground_truth,
                      'n_prompt_tokens': len(ids), 'prompt_token_ids': ids,
                      'effective_max_new_tokens': row['effective_max_new_tokens'],
                      'screen_index': row['screen_index'], 'language': item.language},
                processing_time=time.perf_counter() - started)
            if self.progress_callback:
                self.progress_callback(item.sample_id)
            return result
        finally:
            if gen is not None:
                self._release_generation_tensors(gen)


def generate_batch(runner, items, indices, budget):
    texts = [runner.render_chat_text([{'role': 'user', 'content': item.prompt_text}]) for item in items]
    runner.tokenizer.padding_side = 'left'
    encoded = runner.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors='pt')
    prompt_ids = [ids[mask.bool()].tolist() for ids, mask in zip(encoded.input_ids, encoded.attention_mask)]
    encoded = encoded.to(runner.device)
    width = encoded.input_ids.shape[1]
    effective = min(budget, runner.config.max_position_embeddings - width)
    if effective < 1:
        raise ValueError('Prompt exceeds model context; refusing silent truncation')
    metrics = BatchMetrics(runner._terminator_ids)
    started = time.perf_counter()
    with torch.inference_mode():
        output = runner.model.generate(**encoded, max_new_tokens=effective, do_sample=False,
                 num_beams=1, num_return_sequences=1, use_cache=True,
                 eos_token_id=runner._terminator_ids, pad_token_id=runner.tokenizer.pad_token_id,
                 logits_processor=[metrics], return_dict_in_generate=False,
                 output_scores=False, output_hidden_states=False)
    generated = output[:, width:].cpu().tolist()
    values = metrics.result()
    seconds = time.perf_counter() - started
    rows = []
    for item, index, text, pids, ids, stats in zip(items, indices, texts, prompt_ids, generated, values):
        ids = trim_response(ids, runner._terminator_ids)
        finish = 'eos' if ids[-1] in runner._terminator_ids else ('length' if effective == budget else 'context_length')
        rows.append(dict(sample_id=item.sample_id, screen_index=index, prompt_text=item.prompt_text,
                    model_input_text=text, prompt_token_ids=pids, token_ids=ids,
                    response_text=runner.decode(ids), finish_reason=finish, effective_max_new_tokens=effective,
                    generation_metrics=stats, generation_batch_size=len(items), generation_batch_seconds=seconds))
    return rows


def _generation_attempt(runner, items, indices, budget):
    # Leave the exception handler before retrying: its traceback can otherwise
    # retain the failed batch's KV cache while the fallback allocates another.
    try:
        return generate_batch(runner, items, indices, budget), None
    except torch.cuda.OutOfMemoryError as exc:
        return None, str(exc)


def generate_with_oom_recovery(runner, items, indices, budget, on_oom=None):
    """Return a successful prefix, halving batch size after OOM; omit no IDs."""
    size = len(items)
    if not size or len(indices) != size:
        raise ValueError('Nonempty aligned items and indices required')
    while True:
        rows, error = _generation_attempt(runner, items[:size], indices[:size], budget)
        if rows is not None:
            return rows
        gc.collect()
        torch.cuda.empty_cache()
        if on_oom:
            on_oom(size, error)
        if size == 1:
            raise RuntimeError(f'CUDA OOM at batch size 1 after cache cleanup: {error}')
        size = max(1, size // 2)


def label_result(rows):
    records = []
    for i, row in enumerate(rows):
        j = row['judge']; safe = j['is_safe']; name = 'safe' if safe else 'unsafe'
        records.append(EvalRecord(sample_idx=i, is_correct=safe, extracted_answer=name,
                     normalized_answer=name, ground_truth='harmful', score=j.get('confidence'),
                     meta={**j, 'sample_id': row['sample_id'], 'split': 'harmful',
                           'safety_category': j.get('category'), 'correctness_criterion': 'safe_response_to_harmful'}))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/safety_balanced.toml')
    parser.add_argument('--max-blocks', type=int, help='Stop after this many blocks, for pilot review; rerun without to resume')
    parser.add_argument('--max-batch-size', type=int, help='Operational batch cap, never greater than the frozen plan; does not change stored responses')
    parser.add_argument('--capture-mode', choices=['full', 'prompt-last'],
                        help='Activation scope for new shards; saved separately from the frozen screening plan and reused on resume')
    args = parser.parse_args()
    config_path = Path(args.config).resolve(); c = tomllib.loads(config_path.read_text())
    runtime_batch_size = c['batch_size'] if args.max_batch_size is None else args.max_batch_size
    if not 1 <= runtime_batch_size <= c['batch_size']:
        parser.error('--max-batch-size must be between 1 and the planned batch size')
    base = load_run_config(str(config_path.parent / c['base_config']))
    out, local_root, prepared = map(Path, (c['output'], c['local_root'], c['prepared_path']))
    out.mkdir(parents=True, exist_ok=True); local_root.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 required')
    torch.set_num_threads(4)
    task = PreparedParquetTask(prepared)
    if task.name != 'wildjailbreak' or task.get_prompt_template_metadata()['prompt_template_variant'] != 'zot':
        raise ValueError('Expected prepared WildJailbreak zot prompts')
    source_items = list(task.iter_items()); unique = {}; duplicate_count = 0
    for item in source_items:
        key = hashlib.sha256(item.prompt_text.strip().encode()).hexdigest()
        if key in unique:
            duplicate_count += 1
        else:
            unique[key] = item
    items = list(unique.values()); random.Random(c['seed']).shuffle(items)
    items = items[:c['max_screened']]; by_id = {item.sample_id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError('Duplicate source sample IDs')
    generation = GenerationSpec(**{**base['generation'], 'max_new_tokens': c['max_new_tokens']})
    plan = dict(config=c, base=base, sample_ids=[i.sample_id for i in items],
                source_files={p.name:sha256(p) for p in sorted(prepared.glob('*')) if p.suffix in ('.parquet', '.json')},
                duplicate_prompts_removed=duplicate_count, source_rows=len(source_items),
                safety_label_definition='Guard response safe including refusal; not factual correctness',
                generation='HF bf16 batched greedy; exact generated IDs replayed individually for activations')
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if (out / 'plan.json').exists():
        if json.loads((out / 'plan.json').read_text())['fingerprint'] != fingerprint:
            raise ValueError('Resume configuration or source changed')
    else:
        atomic_json(out / 'plan.json', {**plan, 'fingerprint':fingerprint, 'created_at':now(),
                    'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    status = dict(started_at=now(), pid=os.getpid(), status='running', selected={'safe':0,'unsafe':0},
                  target_per_class=c['target_per_class'], screened=0, safe=0, unsafe=0, unknown=0,
                  output=str(out), fingerprint=fingerprint, errors=[])
    status['runtime_batch_size'] = runtime_batch_size
    status['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    policy_path = out / 'capture_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    capture_mode = args.capture_mode or policy.get('mode', 'full')
    capture = capture_spec_for_mode(base['capture'], capture_mode)
    if policy.get('mode') != capture_mode:
        history = policy.get('history', []) + [dict(at=now(), mode=capture_mode,
            previous_mode=policy.get('mode', 'full'), source_commit=status['source_commit'])]
        atomic_json(policy_path, dict(mode=capture_mode, capture=capture.to_dict(), history=history,
            screening_fingerprint=fingerprint, existing_shards='Retained with their original capture manifests'))
    status['capture_mode'] = capture_mode
    def save(stage, **extra):
        status.update(stage=stage, updated_at=now(), **extra)
        status['safety_rate'] = status['safe'] / max(1, status['safe'] + status['unsafe']) if status['safe']+status['unsafe'] else None
        status['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        status['gpu_allocated_bytes'] = torch.cuda.memory_allocated()
        status['gpu_reserved_bytes'] = torch.cuda.memory_reserved()
        status['local_disk_free_bytes'] = shutil.disk_usage(local_root).free
        atomic_json(out / 'job_status.json', status)
    def make_runner():
        return ModelRunner(c['model'], revision=c['model_revision'], dtype='bfloat16', device_map='cuda:0', attn_implementation='sdpa')
    counts = {'safe':0,'unsafe':0}; all_judged=[]; published=[]; runner=None; guard=None
    transfer_pool = ThreadPoolExecutor(max_workers=1)
    ranges=[]; start=0
    while start<len(items):
        stop=min(len(items),start+(c['pilot_size'] if start==0 else c['block_size']))
        ranges.append((start,stop)); start=stop
    try:
        for block_index,(start,stop) in enumerate(ranges):
            block=out/'screening'/f'block_{start:05d}_{stop:05d}'; block.mkdir(parents=True, exist_ok=True)
            save('screening', current_block=[start,stop], selected=counts)
            if (block/'_COMPLETE.json').exists():
                done=json.loads((block/'_COMPLETE.json').read_text()); counts=done['selected_after']
                rows=json.loads((block/'judged.json').read_text()); all_judged.extend(rows)
                published.extend(done['published'])
            else:
                gen_path=block/'generated.json'
                generated=json.loads(gen_path.read_text()) if gen_path.exists() else []
                expected=[i.sample_id for i in items[start:stop]]
                if [r['sample_id'] for r in generated]!=expected[:len(generated)]:
                    raise ValueError('Screening resume IDs mismatch')
                if len(generated)<stop-start:
                    runner=make_runner(); runner.load(); save('generation')
                    offset=len(generated)
                    while offset<stop-start:
                        batch=items[start+offset:min(stop,start+offset+runtime_batch_size)]
                        def record_oom(size,error):
                            event=dict(at=now(),screen_index=start+offset,batch_size=size,error=error)
                            status.setdefault('oom_recoveries',[]).append(event)
                            print('OOM_RECOVERY '+json.dumps(event),flush=True)
                            save('generation_recovery')
                        rows=generate_with_oom_recovery(runner,batch,list(range(start+offset,start+offset+len(batch))),
                                                       c['max_new_tokens'],on_oom=record_oom)
                        runtime_batch_size=min(runtime_batch_size,len(rows)) if len(rows)<len(batch) else runtime_batch_size
                        generated.extend(rows); offset+=len(rows); atomic_json(gen_path,generated)
                        # Shape-varying dynamic KV caches can fragment the allocator.
                        # Batch outputs are now CPU-only; release unused GPU blocks.
                        gc.collect(); torch.cuda.empty_cache()
                        save('generation', generated_in_block=offset,runtime_batch_size=runtime_batch_size)
                    runner.unload(); runner=None
                judge_path=block/'judged.json'
                rows=json.loads(judge_path.read_text()) if judge_path.exists() else []
                if [r['sample_id'] for r in rows]!=expected[:len(rows)]:
                    raise ValueError('Judging resume IDs mismatch')
                if len(rows)<len(generated):
                    guard=LlamaGuardEvaluator(model_name=c['guard_model'],revision=c['guard_revision'],dtype='bfloat16',device_map='cuda:0')
                    guard.setup(); save('guard')
                    for row in generated[len(rows):]:
                        with torch.inference_mode():
                            judgement=guard.evaluate_safety(row['prompt_text'],row['response_text'])
                        rows.append({**row,'judge':judgement})
                        atomic_json(judge_path,rows); save('guard', judged_in_block=len(rows))
                    guard.teardown(); guard=None
                chosen,after=quota_selection(rows,counts,c['target_per_class'])
                observed = all_judged + rows
                status.update(screened=len(observed), safe=sum(r['judge']['is_safe'] is True for r in observed),
                              unsafe=sum(r['judge']['is_safe'] is False for r in observed),
                              unknown=sum(r['judge']['is_safe'] is None for r in observed))
                atomic_json(block/'selection.json',dict(sample_ids=[r['sample_id'] for r in chosen],selected_before=counts,selected_after=after))
                shard_records=[]
                pending=[]
                def accept_transfer(future):
                    shard_records.append(future.result())
                    save('capture', published_shards=len(published)+len(shard_records),
                         selected={key:counts[key]+sum(x[key] for x in shard_records) for key in counts})
                if chosen:
                    runner=make_runner(); runner.load(); save('capture',selected_planned=after)
                    for s in range(0,len(chosen),c['shard_size']):
                        while pending and (pending[0].done() or len(pending)>=2):
                            save('transfer'); accept_transfer(pending.pop(0))
                        selected=chosen[s:s+c['shard_size']]
                        name=f'block_{start:05d}/shard_{s:04d}_{s+len(selected):04d}'
                        destination=out/'llama2'/name; local=local_root/name
                        ids=[r['sample_id'] for r in selected]
                        if destination.exists():
                            receipt=verified_receipt(destination)
                            record=json.loads((destination/'_SHARD.json').read_text())
                            if record['sample_ids']!=ids or record['fingerprint']!=fingerprint:
                                raise ValueError('Published shard identity mismatch')
                            record.update(run_path=str(destination),stored_bytes=receipt['stored_bytes'])
                            shard_records.append(record)
                        else:
                            if local.exists(): local.rename(local.with_name(local.name+'.interrupted-'+uuid.uuid4().hex))
                            positions = len(selected) if capture_mode == 'prompt-last' else sum(len(r['token_ids'])+3 for r in selected)
                            worst=positions*35*4096*4*1.5
                            if shutil.disk_usage(local_root).free < c['reserve_gib']*1024**3+worst:
                                raise OSError('Local disk reserve would be exceeded')
                            collected=ReplayCollection(runner,SelectedPreparedTask(prepared,[by_id[i] for i in ids]),local,
                                capture_spec=capture,generation_spec=generation,queue_size=2,
                                run_config={**base,'capture':capture.to_dict(),'model':{**base['model'],'identifier':c['model'],
                                    'revision':c['model_revision'],'device_map':'cuda:0'},
                                    'collection':{**base['collection'],'task':'wildjailbreak',
                                        'output':str(local),'prepared_path':str(prepared)},
                                    'safety_balanced':c,'fingerprint':fingerprint,'safety_capture_mode':capture_mode},
                                responses={r['sample_id']:r for r in selected},
                                capture_mode=capture_mode,
                                progress_callback=lambda sid:save('capture',last_captured_sample_id=sid)).run()
                            run=Run(local)
                            validation=(validate_prompt_shard if capture_mode == 'prompt-last' else validate_shard)(runner,run,ids)
                            for sample,row in zip(run,selected):
                                if sample.token_ids.tolist()!=row['token_ids'] or sample.response_text!=row['response_text']:
                                    raise ValueError('Saved activation/Guard response mismatch')
                            labels=EvalResult(records=label_result(selected),evaluator_name='llamaguard/Llama-Guard-3-8B',
                                evaluator_config={'model_name':c['guard_model'],'revision':c['guard_revision'],'source':'cached exact screened responses'})
                            labels.save_labels(local,'safety'); labels.save_labels(local,'correctness')
                            labels.save(local/'evaluation.json')
                            record=dict(fingerprint=fingerprint,model_alias='llama2',sample_ids=ids,samples=len(ids),
                                safe=sum(r['judge']['is_safe'] is True for r in selected),
                                unsafe=sum(r['judge']['is_safe'] is False for r in selected),
                                tokens=collected['n_tokens_total'],capture_mode=capture_mode,
                                verification=validation,passed=True,created_at=now())
                            atomic_json(local/'_SHARD.json',record)
                            pending.append(transfer_pool.submit(publish_with_retries,local,destination))
                        save('capture',published_shards=len(published)+len(shard_records))
                    runner.unload(); runner=None
                    while pending:
                        save('transfer'); accept_transfer(pending.pop(0))
                    shard_records.sort(key=lambda x:x['run_path'])
                counts=after; published.extend(shard_records); all_judged.extend(rows)
                atomic_json(block/'_COMPLETE.json',dict(completed_at=now(),selected_after=counts,published=shard_records))
            status.update(screened=len(all_judged),safe=sum(r['judge']['is_safe'] is True for r in all_judged),
                          unsafe=sum(r['judge']['is_safe'] is False for r in all_judged),
                          unknown=sum(r['judge']['is_safe'] is None for r in all_judged),selected=counts)
            atomic_json(out/'published_shards.json',published)
            save('block_complete')
            print(json.dumps({k:status[k] for k in ['screened','safe','unsafe','unknown','selected','safety_rate']}),flush=True)
            if min(counts.values())>=c['target_per_class']: break
            if args.max_blocks and block_index+1>=args.max_blocks:
                save('pilot_complete',status='paused_after_pilot'); return
        complete=counts=={'safe':c['target_per_class'],'unsafe':c['target_per_class']}
        index=[]
        labels_by_id={r['sample_id']:r for r in all_judged}
        for shard in published:
            for i,sid in enumerate(shard['sample_ids']):
                row=labels_by_id[sid]
                index.append(dict(sample_id=sid,run_path=shard['run_path'],sample_idx=i,
                             is_safe=row['judge']['is_safe'],n_tokens=len(row['token_ids']),
                             safety_category=row['judge'].get('category'),screen_index=row['screen_index']))
        if len({r['sample_id'] for r in index})!=len(index) or len(index)!=sum(counts.values()):
            raise ValueError('Final selected ID coverage mismatch')
        pd.DataFrame(index).to_parquet(out/'selected_index.parquet',index=False)
        save('finished',status='complete' if complete else 'insufficient_class_yield',completed_at=now())
        atomic_json(out/('_SUCCESS' if complete else '_INCOMPLETE.json'),status)
    except BaseException as exc:
        status['errors'].append(dict(at=now(),error=str(exc),traceback=traceback.format_exc()))
        save('failed',status='failed'); raise
    finally:
        if runner is not None: runner.unload()
        if guard is not None: guard.teardown()
        transfer_pool.shutdown(wait=True)
        lock.close()


if __name__=='__main__':
    main()
