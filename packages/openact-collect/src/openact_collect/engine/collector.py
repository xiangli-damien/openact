import gc
import json
import logging
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from openact_collect.engine.async_writer import AsyncZarrWriter
from openact_collect.engine.completion_marker import CompletionMarker
from openact_collect.engine.model_manager import GenerationResult, ModelRunner
from openact_collect.engine.offset_calculator import OffsetCalculator
from openact_collect.engine.online_metrics import OnlineMetricsProcessor
from openact_collect.extractors.base import Extractor
from openact_collect.extractors.hidden_state_data import GenerationMetrics, HiddenStateData
from openact_collect.extractors.hidden_state_extractor import HiddenStateExtractor
from openact_collect.schema import (
    CaptureSpec,
    GenerationProfile,
    GenerationSpec,
    SafetySpec,
    StorageSpec,
)
from openact_collect.tasks.base import Task, TaskItem
from openact_collect.tasks.safety.base import SafetyTaskItem
from openact_core.schema.manifest import (
    DatasetSpec,
    EnvironmentSpec,
    Manifest,
    ModelSpec,
    PromptSpec,
    StatsSpec,
)
from openact_core.schema.status import SampleStatus

logger = logging.getLogger("openact.collect")

PARQUET_BATCH_ROWS = 5000


@dataclass
class CollectResult:
    sample_idx: int
    status: SampleStatus
    response_text: str = ""
    token_ids: List[int] = field(default_factory=list)
    token_offsets: Optional[np.ndarray] = None
    finish_reason: str = "unknown"
    hidden_state_data: Optional[HiddenStateData] = None
    generation_metrics: Optional[GenerationMetrics] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None
    processing_time: float = 0.0


class CollectionRunner:

    def __init__(
        self,
        model_manager: ModelRunner,
        task: Task,
        output_dir: Union[str, Path],
        capture_spec: Optional[CaptureSpec] = None,
        generation_spec: Optional[GenerationSpec] = None,
        storage_spec: Optional[StorageSpec] = None,
        queue_size: int = 8,
    ):
        self.model_manager = model_manager
        self.task = task
        self.output_dir = Path(output_dir)
        self.capture_spec = capture_spec or CaptureSpec()
        self.generation_spec = generation_spec or GenerationSpec()
        self.storage_spec = storage_spec or StorageSpec()
        self.queue_size = queue_size

        self._profiles = task.get_profiles()
        self.manifest: Optional[Manifest] = None
        self.writer: Optional[AsyncZarrWriter] = None
        self.extractor: Optional[Extractor] = None
        self.offset_calculator: Optional[OffsetCalculator] = None
        self._parquet_buffer: List[Dict[str, Any]] = []
        self._parquet_part_paths: List[Path] = []
        self._interrupted = False

    def __enter__(self) -> "CollectionRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if self.writer is not None:
                self._flush_and_merge_parquet()
                self.writer.finalize()
        except Exception as cleanup_err:
            logger.error("Error during CollectionRunner cleanup: %s", cleanup_err)
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

        self.offset_calculator = OffsetCalculator(
            self.model_manager.tokenizer,
            special_ids=self.model_manager.get_special_ids(),
        )
        self.extractor = HiddenStateExtractor(self.capture_spec, self.model_manager)

        self.writer = AsyncZarrWriter(
            zarr_path=self.output_dir / "tensors.zarr",
            manifest=self.manifest,
            capture_spec=self.capture_spec,
            storage_spec=self.storage_spec,
            queue_size=self.queue_size,
        )

        stats: Dict[str, Any] = {
            "total": total,
            "processed": 0,
            "ok": 0,
            "error": 0,
            "timeout": 0,
            "skipped": 0,
            "oom": 0,
            "start_time": time.time(),
        }
        self.manifest.stats.start_time = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        try:
            self.writer.start()
            with tqdm(total=total, desc="Collecting") as pbar:
                for item in items_iter:
                    result = self._process_safe(item)

                    if result.status == SampleStatus.OK:
                        self.writer.submit_sample(
                            sample_idx=result.sample_idx,
                            token_ids=np.array(result.token_ids, dtype=np.int32),
                            token_offsets=result.token_offsets,
                            hidden_state_data=result.hidden_state_data,
                            status=result.status,
                        )
                    else:
                        self.writer.mark_failed(result.sample_idx, result.status)

                    self._buffer_parquet_row(result, item)

                    stats["processed"] += 1
                    status_key = result.status.name.lower()
                    stats[status_key] = stats.get(status_key, 0) + 1

                    acc = (
                        stats["ok"] / stats["processed"] * 100
                        if stats["processed"] > 0
                        else 0
                    )
                    pbar.set_postfix(
                        {
                            "OK": stats["ok"],
                            "Err": stats["error"] + stats["timeout"],
                            "Acc": f"{acc:.1f}%",
                        }
                    )
                    pbar.update(1)

                    if stats["processed"] % 100 == 0:
                        self._force_memory_cleanup()

        except KeyboardInterrupt:
            logger.warning("Collection interrupted by user")
            self._interrupted = True
            raise
        except Exception as e:
            logger.exception("Collection failed: %s", e)
            self._interrupted = True
            raise
        finally:
            self._flush_and_merge_parquet()
            n_tokens_total = (
                self.writer.current_token_count if self.writer is not None else 0
            )
            if self.writer is not None:
                self.writer.finalize()

            stats["end_time"] = time.time()
            stats["duration_seconds"] = stats["end_time"] - stats["start_time"]

            self.manifest.stats.n_samples_total = stats["processed"]
            self.manifest.stats.n_samples_ok = stats["ok"]
            self.manifest.stats.n_samples_error = stats.get("error", 0) + stats.get(
                "timeout", 0
            )
            self.manifest.stats.n_tokens_total = n_tokens_total
            stats["n_tokens_total"] = n_tokens_total
            self.manifest.stats.end_time = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            self.manifest.stats.duration_seconds = stats["duration_seconds"]
            self.manifest.save(self.output_dir / "manifest.json")

        if not self._interrupted:
            self._finalize(stats)

        return stats

    def collect_batch(self, items: Sequence[TaskItem]) -> List[CollectResult]:
        results: List[CollectResult] = []
        for item in items:
            result = self._process_safe(item)
            results.append(result)
        return results

    def _buffer_parquet_row(self, result: CollectResult, item: TaskItem) -> None:
        row = {
            "sample_idx": result.sample_idx,
            "sample_id": result.meta.get("sample_id", str(result.sample_idx)),
            "status": int(result.status),
            "prompt_text": item.prompt_text,
            "response_text": result.response_text,
            "ground_truth": item.ground_truth,
            "finish_reason": result.finish_reason,
            "n_prompt_tokens": result.meta.get("n_prompt_tokens", 0),
            "n_response_tokens": len(result.token_ids),
            "processing_time": result.processing_time,
            "error_msg": result.error_msg,
            "max_probability": (
                result.generation_metrics.max_probability
                if result.generation_metrics is not None
                else None
            ),
            "perplexity": (
                result.generation_metrics.perplexity
                if result.generation_metrics is not None
                else None
            ),
            "entropy": (
                result.generation_metrics.entropy
                if result.generation_metrics is not None
                else None
            ),
            "n_hidden_state_tokens": (
                result.hidden_state_data.n_tokens
                if result.hidden_state_data is not None
                else None
            ),
        }

        parquet_meta = item.parquet_meta()
        for k, v in parquet_meta.items():
            if k not in row:
                row[k] = v

        self._parquet_buffer.append(row)
        if len(self._parquet_buffer) >= PARQUET_BATCH_ROWS:
            self._flush_parquet_buffer()

    def _flush_parquet_buffer(self) -> None:
        if not self._parquet_buffer:
            return
        part_path = (
            self.output_dir / f"_part_{len(self._parquet_part_paths):04d}.parquet"
        )
        df = pd.DataFrame(self._parquet_buffer)
        df.to_parquet(part_path, index=False)
        self._parquet_part_paths.append(part_path)
        self._parquet_buffer.clear()

    def _flush_and_merge_parquet(self) -> None:
        self._flush_parquet_buffer()
        if not self._parquet_part_paths:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        tables = []
        for p in self._parquet_part_paths:
            if p.exists():
                tables.append(pq.read_table(p))
        if tables:
            merged = pa.concat_tables(tables)
            pq.write_table(merged, self.output_dir / "data.parquet")
        for p in self._parquet_part_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def _finalize(self, stats: Dict[str, Any]) -> None:
        logger.info("Finalizing run...")
        CompletionMarker.mark_complete(
            run_dir=self.output_dir,
            stats={
                "n_samples_total": stats.get("processed", 0),
                "n_samples_ok": stats.get("ok", 0),
                "n_samples_error": stats.get("error", 0) + stats.get("timeout", 0),
                "n_tokens_total": (
                    self.writer.current_token_count if self.writer else 0
                ),
                "duration_seconds": stats.get("duration_seconds", 0),
            },
        )
        logger.info("Run completed: %s", self.output_dir)

    def _create_manifest(self) -> Manifest:
        model_spec = self.model_manager.get_model_spec()
        prompt_template = self.task.get_prompt_template()
        prompt_spec = PromptSpec(
            template_name=prompt_template.name,
            template_text=prompt_template.template,
            variables=prompt_template.variables,
        )
        dataset_spec = DatasetSpec(
            name=self.task.name,
            source=self.task.source,
            split=self.task.split,
            language=self.task.language,
            max_samples=self.task.max_samples,
        )

        import openact_collect

        env_spec = EnvironmentSpec(
            openact_version=openact_collect.__version__,
            python_version=platform.python_version(),
            torch_version=torch.__version__,
        )
        try:
            import transformers

            env_spec.transformers_version = transformers.__version__
        except ImportError:
            pass
        if torch.cuda.is_available():
            env_spec.cuda_version = torch.version.cuda
        env_spec.hostname = platform.node()
        env_spec.command_line = " ".join(sys.argv)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.task.name}_{model_spec.name}_{timestamp}"

        manifest = Manifest(
            run_id=run_id,
            model=model_spec,
            dataset=dataset_spec,
            prompt=prompt_spec,
            generation_config=self.generation_spec.to_dict(),
            capture_config=self.capture_spec.to_dict(),
            storage_config=self.storage_spec.to_dict(),
            environment=env_spec,
        )

        safety_spec = self.task.get_safety_spec()
        if safety_spec is not None:
            manifest.custom["safety"] = safety_spec.to_dict()

        return manifest

    def _process_safe(self, item: TaskItem) -> CollectResult:
        try:
            return self._process_sample(item)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM on sample %d, skipping", item.sample_idx)
            self._force_memory_cleanup()
            return CollectResult(
                sample_idx=item.sample_idx,
                status=SampleStatus.ERROR,
                meta={"sample_id": item.sample_id},
                error_msg="CUDA OOM",
                processing_time=0.0,
            )
        except Exception:
            logger.exception("Failed on sample %d", item.sample_idx)
            return CollectResult(
                sample_idx=item.sample_idx,
                status=SampleStatus.ERROR,
                meta={"sample_id": item.sample_id},
                error_msg=traceback.format_exc()[-500:],
                processing_time=0.0,
            )

    def _process_sample(self, item: TaskItem) -> CollectResult:
        start_time = time.time()
        gen_result = None

        try:
            messages = [{"role": "user", "content": item.prompt_text}]
            input_ids = self.model_manager.apply_chat_template(messages)

            if isinstance(item, SafetyTaskItem):
                profile_name = getattr(item, "profile", None)
            else:
                profile_name = item.meta.get("profile")

            seed_override = item.meta.get("seed")

            if (
                profile_name
                and self._profiles
                and profile_name in self._profiles
            ):
                profile = self._profiles[profile_name]
                if seed_override is not None:
                    from dataclasses import replace

                    effective_gen_spec = replace(profile, seed=seed_override)
                else:
                    effective_gen_spec = profile
            else:
                effective_gen_spec = self.generation_spec

            metrics_processor = None
            logits_processors = None
            if self.capture_spec.compute_online_metrics:
                is_greedy = (
                    effective_gen_spec.temperature == 0
                    or not effective_gen_spec.do_sample
                )
                metrics_processor = OnlineMetricsProcessor(greedy=is_greedy)
                logits_processors = [metrics_processor]

            gen_result = self.model_manager.generate(
                input_ids=input_ids,
                gen_spec=effective_gen_spec,
                logits_processors=logits_processors,
                capture_hidden_states=bool(self.capture_spec.hidden_states),
            )

            hidden_state_data = self.extractor.extract(gen_result)

            if (
                self.capture_spec.hidden_states
                and gen_result.keep_indices
                and len(gen_result.keep_indices) > 0
                and hidden_state_data.n_tokens == 0
            ):
                self._release_generation_tensors(gen_result)
                return CollectResult(
                    sample_idx=item.sample_idx,
                    status=SampleStatus.ERROR,
                    response_text=self.model_manager.decode(
                        gen_result.token_ids, skip_special_tokens=True
                    ),
                    token_ids=gen_result.token_ids,
                    finish_reason=gen_result.finish_reason,
                    meta={
                        "sample_id": item.sample_id,
                        "prompt_text": item.prompt_text,
                        "ground_truth": item.ground_truth,
                        "n_prompt_tokens": gen_result.input_length,
                        "language": getattr(item, "language", "en"),
                        **item.meta,
                    },
                    error_msg="Hidden state extraction failed: keep_indices out of bounds",
                    processing_time=time.time() - start_time,
                )

            if (
                self.capture_spec.hidden_states
                and self.capture_spec.save_prompt_last
            ):
                prompt_last = self.extractor.extract_prompt_last(input_ids)
                hidden_state_data.prompt_last_states = prompt_last

            self._release_generation_tensors(gen_result)

            generation_metrics = None
            if metrics_processor is not None:
                generation_metrics = metrics_processor.finalize()

            response_text = self.model_manager.decode(
                gen_result.token_ids, skip_special_tokens=True
            )

            token_offsets = self.offset_calculator.compute_offsets(
                text=response_text, token_ids=gen_result.token_ids
            )

            processing_time = time.time() - start_time

            return CollectResult(
                sample_idx=item.sample_idx,
                status=SampleStatus.OK,
                response_text=response_text,
                token_ids=gen_result.token_ids,
                token_offsets=token_offsets,
                finish_reason=gen_result.finish_reason,
                hidden_state_data=hidden_state_data,
                generation_metrics=generation_metrics,
                meta={
                    "sample_id": item.sample_id,
                    "prompt_text": item.prompt_text,
                    "ground_truth": item.ground_truth,
                    "n_prompt_tokens": gen_result.input_length,
                    "language": getattr(item, "language", "en"),
                    **item.meta,
                },
                processing_time=processing_time,
            )

        except TimeoutError:
            if gen_result is not None:
                self._release_generation_tensors(gen_result)
            return CollectResult(
                sample_idx=item.sample_idx,
                status=SampleStatus.TIMEOUT,
                meta={"sample_id": item.sample_id},
                error_msg="Timeout",
                processing_time=time.time() - start_time,
            )
        finally:
            if gen_result is not None:
                self._release_generation_tensors(gen_result)

    def _release_generation_tensors(self, gen_result: GenerationResult) -> None:
        if gen_result is None:
            return
        if gen_result.hidden_states is not None:
            try:
                for step_hs in gen_result.hidden_states:
                    if isinstance(step_hs, (list, tuple)):
                        for t in step_hs:
                            if isinstance(t, torch.Tensor):
                                del t
                del gen_result.hidden_states
            except Exception:
                pass
            gen_result.hidden_states = None
        if gen_result.scores is not None:
            try:
                for s in gen_result.scores:
                    if isinstance(s, torch.Tensor):
                        del s
                del gen_result.scores
            except Exception:
                pass
            gen_result.scores = None
        for attr in ("input_ids", "generated_tokens", "full_sequence"):
            if getattr(gen_result, attr, None) is not None:
                delattr(gen_result, attr)
                setattr(gen_result, attr, None)

    def _force_memory_cleanup(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"CollectionRunner(task='{self.task.name}', output='{self.output_dir}')"