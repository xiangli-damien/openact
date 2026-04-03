import gc
import logging
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from openact_collect.engine.async_writer import ZarrWriter
from openact_collect.engine.completion_marker import CompletionMarker
from openact_collect.engine.model_manager import GenerationResult, ModelRunner
from openact_collect.engine.offset_calculator import OffsetCalculator
from openact_collect.engine.online_metrics import OnlineMetricsProcessor
from openact_collect.extractors.base import Extractor
from openact_collect.extractors.hidden_state_data import GenerationMetrics, HiddenStateData
from openact_collect.extractors.hidden_state_extractor import HiddenStateExtractor
from openact_collect.schema import CaptureSpec, GenerationSpec, StorageSpec
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.safety.base import SafetyTaskItem
from openact_core.schema.manifest import DatasetSpec, EnvironmentSpec, Manifest, PromptSpec
from openact_core.schema.status import SampleStatus

logger = logging.getLogger('openact.collect')
PARQUET_BATCH_ROWS = 5000


@dataclass
class CollectResult:
    sample_idx: int
    status: SampleStatus
    response_text: str = ''
    token_ids: List[int] = field(default_factory=list)
    token_offsets: Optional[np.ndarray] = None
    finish_reason: str = 'unknown'
    hidden_state_data: Optional[HiddenStateData] = None
    generation_metrics: Optional[GenerationMetrics] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None
    processing_time: float = 0.0


class CollectionRunner:
    def __init__(self, model_manager: ModelRunner, task: Task, output_dir: Union[str, Path], capture_spec: Optional[CaptureSpec] = None, generation_spec: Optional[GenerationSpec] = None, storage_spec: Optional[StorageSpec] = None, queue_size: int = 8):
        self.model_manager = model_manager
        self.task = task
        self.output_dir = Path(output_dir)
        self.capture_spec = capture_spec or CaptureSpec()
        self.generation_spec = generation_spec or GenerationSpec()
        self.storage_spec = storage_spec or StorageSpec()
        self.queue_size = queue_size
        self._profiles = task.get_profiles()
        self.manifest: Optional[Manifest] = None
        self.writer: Optional[ZarrWriter] = None
        self.extractor: Optional[Extractor] = None
        self.offset_calculator: Optional[OffsetCalculator] = None
        self._parquet_buffer: List[Dict[str, Any]] = []
        self._parquet_part_paths: List[Path] = []
        self._interrupted = False

    def __enter__(self) -> 'CollectionRunner':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def _resolve_plan(self):
        estimated = self.task.estimate_size()
        if estimated is not None:
            return estimated, self.task.iter_items()
        plan = list(self.task.iter_items())
        return len(plan), iter(plan)

    def run(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_manager._loaded:
            self.model_manager.load()
        total, items_iter = self._resolve_plan()
        self.manifest = self._create_manifest()
        self.manifest.dataset.n_samples = total
        self.offset_calculator = OffsetCalculator(self.model_manager.tokenizer, special_ids=self.model_manager.get_special_ids())
        self.extractor = HiddenStateExtractor(self.capture_spec, self.model_manager)
        self.writer = ZarrWriter(zarr_path=self.output_dir / 'tensors.zarr', manifest=self.manifest, capture_spec=self.capture_spec, storage_spec=self.storage_spec, queue_size=self.queue_size)
        stats: Dict[str, Any] = {
            'total': total,
            'processed': 0,
            'ok': 0,
            'error': 0,
            'timeout': 0,
            'skipped': 0,
            'oom': 0,
            'start_time': time.time(),
        }
        self.manifest.stats.start_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        try:
            self.writer.start()
            with tqdm(total=total, desc='Collecting') as progress:
                for item in items_iter:
                    result = self._process_safe(item)
                    if result.status == SampleStatus.OK:
                        self.writer.submit_sample(
                            sample_idx=result.sample_idx,
                            token_ids=np.asarray(result.token_ids, dtype=np.int32),
                            token_offsets=result.token_offsets if result.token_offsets is not None else np.zeros((0, 2), dtype=np.int32),
                            hidden_state_data=result.hidden_state_data,
                            status=result.status,
                        )
                    else:
                        self.writer.mark_failed(result.sample_idx, result.status)
                    self._buffer_parquet_row(result, item)
                    stats['processed'] += 1
                    stats[result.status.name.lower()] = stats.get(result.status.name.lower(), 0) + 1
                    ok = stats['ok']
                    processed = stats['processed']
                    progress.set_postfix({'OK': ok, 'Err': stats['error'] + stats['timeout'], 'Acc': f'{(100 * ok / processed if processed else 0):.1f}%'} )
                    progress.update(1)
        except KeyboardInterrupt:
            logger.warning('Collection interrupted by user')
            self._interrupted = True
            raise
        except Exception as exc:
            logger.exception('Collection failed: %s', exc)
            self._interrupted = True
            raise
        finally:
            self._flush_and_merge_parquet()
            n_tokens_total = self.writer.current_token_count if self.writer is not None else 0
            if self.writer is not None:
                self.writer.finalize()
            stats['end_time'] = time.time()
            stats['duration_seconds'] = stats['end_time'] - stats['start_time']
            stats['n_tokens_total'] = n_tokens_total
            self.manifest.stats.n_samples_total = stats['processed']
            self.manifest.stats.n_samples_ok = stats['ok']
            self.manifest.stats.n_samples_error = stats['error'] + stats['timeout']
            self.manifest.stats.n_tokens_total = n_tokens_total
            self.manifest.stats.end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            self.manifest.stats.duration_seconds = stats['duration_seconds']
            self.manifest.save(self.output_dir / 'manifest.json')
        if not self._interrupted:
            self._finalize(stats)
        return stats

    def collect_batch(self, items: Sequence[TaskItem]) -> List[CollectResult]:
        return [self._process_safe(item) for item in items]

    def _buffer_parquet_row(self, result: CollectResult, item: TaskItem) -> None:
        import json
        semantic_meta = item.semantic_meta()
        row: Dict[str, Any] = {
            'sample_idx': result.sample_idx,
            'sample_id': result.meta.get('sample_id', str(result.sample_idx)),
            'status': int(result.status),
            'prompt_text': result.meta.get('prompt_text'),
            'prompt_fields_json': json.dumps(item.prompt_fields, ensure_ascii=False) if getattr(item, 'prompt_fields', None) else None,
            'response_text': result.response_text,
            'ground_truth': item.ground_truth,
            'finish_reason': result.finish_reason,
            'n_prompt_tokens': result.meta.get('n_prompt_tokens', 0),
            'n_response_tokens': len(result.token_ids),
            'processing_time': result.processing_time,
            'error_msg': result.error_msg,
            'max_probability': None if result.generation_metrics is None else result.generation_metrics.max_probability,
            'perplexity': None if result.generation_metrics is None else result.generation_metrics.perplexity,
            'entropy': None if result.generation_metrics is None else result.generation_metrics.entropy,
            'n_hidden_state_tokens': None if result.hidden_state_data is None else result.hidden_state_data.n_tokens,
            'semantic_meta_json': json.dumps(semantic_meta, ensure_ascii=False) if semantic_meta else None,
            **self.task.get_task_metadata(item),
            **self.task.get_prompt_template_metadata(item),
        }
        for key, value in item.parquet_meta().items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            row.setdefault(key, value)
        self._parquet_buffer.append(row)
        if len(self._parquet_buffer) >= PARQUET_BATCH_ROWS:
            self._flush_parquet_buffer()

    def _flush_parquet_buffer(self) -> None:
        if not self._parquet_buffer:
            return
        part_path = self.output_dir / f'_part_{len(self._parquet_part_paths):04d}.parquet'
        pd.DataFrame(self._parquet_buffer).to_parquet(part_path, index=False)
        self._parquet_part_paths.append(part_path)
        self._parquet_buffer.clear()

    def _flush_and_merge_parquet(self) -> None:
        self._flush_parquet_buffer()
        if not self._parquet_part_paths:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        tables = [pq.read_table(path) for path in self._parquet_part_paths if path.exists()]
        if tables:
            pq.write_table(pa.concat_tables(tables), self.output_dir / 'data.parquet')
        for path in self._parquet_part_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _finalize(self, stats: Dict[str, Any]) -> None:
        logger.info('Finalizing run...')
        CompletionMarker.mark_complete(
            run_dir=self.output_dir,
            stats={
                'n_samples_total': stats.get('processed', 0),
                'n_samples_ok': stats.get('ok', 0),
                'n_samples_error': stats.get('error', 0) + stats.get('timeout', 0),
                'n_tokens_total': 0 if self.writer is None else self.writer.current_token_count,
                'duration_seconds': stats.get('duration_seconds', 0.0),
            },
        )
        logger.info('Run completed: %s', self.output_dir)

    def _create_manifest(self) -> Manifest:
        model_spec = self.model_manager.get_model_spec()
        prompt_template = self.task.get_prompt_template()
        import openact_collect
        return Manifest(
            run_id=f"{self.task.name}_{model_spec.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            model=model_spec,
            dataset=DatasetSpec(name=self.task.name, source=self.task.source, split=self.task.split, language=getattr(self.task, 'language', 'en'), max_samples=self.task.max_samples),
            prompt=PromptSpec(
                template_name=prompt_template.name,
                template_text=prompt_template.template,
                variables=list(prompt_template.variables),
                language=prompt_template.language,
                requested_language=prompt_template.requested_language,
                localization_mode=prompt_template.localization_mode,
                answer_prefix=prompt_template.answer_prefix,
                template_variant=getattr(self.task, '_template_variant', None),
                supports_multilingual=prompt_template.supports_multilingual,
            ),
            generation_config=self.generation_spec.to_dict(),
            capture_config=self.capture_spec.to_dict(),
            storage_config=self.storage_spec.to_dict(),
            environment=EnvironmentSpec(
                openact_version=openact_collect.__version__,
                python_version=platform.python_version(),
                torch_version=torch.__version__,
                transformers_version=self._transformers_version(),
                cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
                hostname=platform.node(),
                command_line=' '.join(sys.argv),
            ),
            custom={'safety': self.task.get_safety_spec().to_dict()} if self.task.get_safety_spec() is not None else {},
        )

    @staticmethod
    def _transformers_version() -> Optional[str]:
        try:
            import transformers
            return transformers.__version__
        except ImportError:
            return None

    def _resolve_generation_spec(self, item: TaskItem) -> GenerationSpec:
        profile_name = item.profile if isinstance(item, SafetyTaskItem) else item.meta.get('profile')
        seed_override = item.meta.get('seed')
        if profile_name and self._profiles and profile_name in self._profiles:
            profile = self._profiles[profile_name]
            if seed_override is not None:
                return replace(profile, seed=seed_override)
            return profile
        if seed_override is not None:
            return replace(self.generation_spec, seed=seed_override)
        return self.generation_spec

    def _process_safe(self, item: TaskItem) -> CollectResult:
        rendered_prompt = self.task.render_prompt(item)
        error_meta = {
            'sample_id': item.sample_id,
            'prompt_text': rendered_prompt,
            'ground_truth': item.ground_truth,
            'language': getattr(item, 'language', 'en'),
            **item.meta,
        }
        try:
            return self._process_sample(item, prompt_text=rendered_prompt)
        except torch.cuda.OutOfMemoryError:
            logger.warning('OOM on sample %d, skipping', item.sample_idx)
            self._force_memory_cleanup()
            return CollectResult(sample_idx=item.sample_idx, status=SampleStatus.ERROR, token_offsets=np.zeros((0, 2), dtype=np.int32), meta=error_meta, error_msg='CUDA OOM')
        except Exception:
            logger.exception('Failed on sample %d', item.sample_idx)
            return CollectResult(sample_idx=item.sample_idx, status=SampleStatus.ERROR, token_offsets=np.zeros((0, 2), dtype=np.int32), meta=error_meta, error_msg=traceback.format_exc()[-1000:])

    def _process_sample(self, item: TaskItem, prompt_text: Optional[str] = None) -> CollectResult:
        started_at = time.time()
        gen_result: Optional[GenerationResult] = None
        try:
            prompt_text = prompt_text or self.task.render_prompt(item)
            messages = [{'role': 'user', 'content': prompt_text}]
            input_ids = self.model_manager.apply_chat_template(messages)
            gen_spec = self._resolve_generation_spec(item)
            metrics_processor: Optional[OnlineMetricsProcessor] = None
            logits_processors = None
            if self.capture_spec.compute_online_metrics:
                greedy = gen_spec.temperature == 0 or not gen_spec.do_sample
                metrics_processor = OnlineMetricsProcessor(greedy=greedy)
                logits_processors = [metrics_processor]
            gen_result = self.model_manager.generate(input_ids=input_ids, gen_spec=gen_spec, logits_processors=logits_processors, capture_spec=self.capture_spec)
            response_text = self.model_manager.decode(gen_result.token_ids, skip_special_tokens=True)
            hidden_state_data = self.extractor.extract(gen_result, input_ids=input_ids)
            token_offsets = self.offset_calculator.compute_offsets(response_text, gen_result.token_ids)
            generation_metrics = metrics_processor.finalize() if metrics_processor is not None else None
            meta = {
                'sample_id': item.sample_id,
                'prompt_text': prompt_text,
                'ground_truth': item.ground_truth,
                'n_prompt_tokens': gen_result.input_length,
                'language': getattr(item, 'language', 'en'),
                **item.meta,
            }
            return CollectResult(
                sample_idx=item.sample_idx,
                status=SampleStatus.OK,
                response_text=response_text,
                token_ids=list(gen_result.token_ids),
                token_offsets=token_offsets,
                finish_reason=gen_result.finish_reason,
                hidden_state_data=hidden_state_data,
                generation_metrics=generation_metrics,
                meta=meta,
                processing_time=time.time() - started_at,
            )
        finally:
            self._release_generation_tensors(gen_result)

    def _release_generation_tensors(self, gen_result: Optional[GenerationResult]) -> None:
        if gen_result is None:
            return
        gen_result.hidden_states = []
        gen_result.traces = None
        gen_result.scores = None
        gen_result.input_ids = None
        gen_result.generated_tokens = None
        gen_result.full_sequence = None

    @staticmethod
    def _force_memory_cleanup() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"CollectionRunner(task={self.task.name!r}, output={str(self.output_dir)!r})"
