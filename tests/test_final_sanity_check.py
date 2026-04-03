#!/usr/bin/env python3
"""
Final Sanity Check: Full Pipeline Verification with Real Model
===============================================================

For every CAPABILITY task, this script:
  1. Downloads dataset, takes 3 samples
  2. Renders prompts — verifies no unresolved {placeholders}
  3. Applies chat template — verifies tokenization
  4. Generates with Qwen (max_new_tokens=16, enough to test pipeline)
  5. Extracts hidden states — verifies shape (T, L, H), non-zero
  6. Computes token offsets — verifies monotonicity, coverage
  7. Runs full CollectionRunner — verifies zarr + parquet output integrity

Usage (from OpenAct repo root, with GPU recommended for Phase 2–3):
    export PYTHONPATH=packages/openact-core/src:packages/openact-collect/src:packages/openact-eval/src
    python tests/test_final_sanity_check.py

    # Full run with all capability tasks (~0.5B model needs modest VRAM)
    python tests/test_final_sanity_check.py --model Qwen/Qwen2-0.5B-Instruct

    # Single task
    python tests/test_final_sanity_check.py --task gsm8k

    # Several tasks (mgsm / belebele keep all language variants in CAPABILITY_CONFIGS)
    python tests/test_final_sanity_check.py --tasks math mgsm theoremqa belebele commonsenseqa

    # Use a specific model
    python tests/test_final_sanity_check.py --model Qwen/Qwen2-7B-Instruct

    # Adjust token generation length
    python tests/test_final_sanity_check.py --max-tokens 32

    # Skip Phase 3 (data + model only)
    python tests/test_final_sanity_check.py --skip-collection

    # Phase 3 only for one task
    python tests/test_final_sanity_check.py --collection-only mmlu

Deps: torch, transformers, numpy, zarr, pyarrow, pandas, tqdm, datasets (HF), and network for models/data.
"""

import argparse
import gc
import json
import re
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Repo path setup (works when run as tests/test_final_sanity_check.py) ─
ROOT = Path(__file__).resolve().parent.parent
for pkg in ("openact-core", "openact-collect", "openact-eval"):
    for candidate in [
        ROOT / pkg / "src",
        ROOT.parent / pkg / "src",
        ROOT / "packages" / pkg / "src",
        ROOT.parent / "packages" / pkg / "src",
    ]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break

import numpy as np
import torch

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
MAX_SAMPLES_PER_TASK = 3
MAX_NEW_TOKENS = 16  # just enough to verify pipeline, keep it fast

# All capability tasks with their specific kwargs
CAPABILITY_CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("gsm8k", {}),
    ("mmlu", {}),
    ("math", {}),
    ("theoremqa", {}),
    ("arc_challenge", {}),
    ("commonsenseqa", {}),
    ("truthfulqa", {}),
    ("humaneval", {}),
    ("ifeval", {}),
    # Multilingual: one lang each to verify localization works
    ("mgsm", {"language": "en"}),
    ("mgsm", {"language": "zh"}),
    ("mgsm", {"language": "ja"}),
    ("belebele", {"language": "en"}),
    ("belebele", {"language": "zh"}),
    ("belebele", {"language": "ar"}),
]


# ── Result tracking ──────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str = ""
    error: Optional[str] = None


@dataclass
class TaskReport:
    config_id: str
    task_name: str
    language: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ── Helpers ──────────────────────────────────────────────────────────────


def trunc(s, n=120):
    s = str(s) if s is not None else "None"
    return s[:n] + ("…" if len(s) > n else "")


def section(title: str):
    w = 80
    print(f"\n{'━' * w}")
    print(f"  {title}")
    print(f"{'━' * w}")


# ── Phase 1: Dataset + Prompt Checks (no model needed) ──────────────────


def check_dataset_and_prompts(task_name: str, extra_kwargs: Dict[str, Any]) -> TaskReport:
    """Verify dataset download, item structure, and prompt rendering."""
    from openact_collect.tasks.registry import TaskRegistry
    from openact_core.tasks.templates import ANSWER_PREFIXES

    lang = extra_kwargs.get("language", "en")
    config_id = f"{task_name}_{lang}" if "language" in extra_kwargs else task_name
    report = TaskReport(config_id=config_id, task_name=task_name, language=lang)

    # ── Check 1: Task creation ──
    try:
        task = TaskRegistry.create(task_name, max_samples=MAX_SAMPLES_PER_TASK, **extra_kwargs)
        report.checks.append(
            CheckResult(
                "task_creation",
                True,
                f"source={task.source} split={task.split} template={task._template_variant}",
            )
        )
    except Exception as e:
        report.checks.append(CheckResult("task_creation", False, error=str(e)))
        return report  # Can't continue without task

    # ── Check 2: Items iteration ──
    try:
        items = list(task.iter_items())
        assert len(items) > 0, "No items returned"
        assert len(items) <= MAX_SAMPLES_PER_TASK, f"Got {len(items)} > {MAX_SAMPLES_PER_TASK}"
        report.checks.append(CheckResult("items_iteration", True, f"got {len(items)} items"))
    except Exception as e:
        report.checks.append(CheckResult("items_iteration", False, error=str(e)))
        return report

    # ── Check 3: Item field integrity ──
    for idx, item in enumerate(items):
        try:
            assert item.sample_idx == idx, f"sample_idx={item.sample_idx} != {idx}"
            assert item.sample_id, "Empty sample_id"
            assert isinstance(item.prompt_fields, dict), "prompt_fields not a dict"
            assert len(item.prompt_fields) > 0 or item.prompt_text, "No prompt_fields and no prompt_text"
            assert item.language, "Empty language"

            # Task-specific ground truth format validation
            gt = item.ground_truth
            if task_name in ("gsm8k", "mgsm"):
                assert gt is not None, "Numeric task must have GT"
                float(str(gt).replace(",", ""))  # must be parseable as number
            elif task_name in ("mmlu", "arc_challenge", "commonsenseqa", "belebele"):
                assert gt is not None, "MC task must have GT"
                assert str(gt).strip() in "ABCDE", f"GT not a letter: {gt!r}"
            elif task_name in ("math", "theoremqa", "truthfulqa"):
                assert gt is not None, f"{task_name} must have GT"
            # humaneval, ifeval: GT can be None

            report.checks.append(
                CheckResult(
                    f"item[{idx}]_fields",
                    True,
                    f"id={item.sample_id} GT={trunc(gt, 40)} fields={list(item.prompt_fields.keys())}",
                )
            )
        except Exception as e:
            report.checks.append(CheckResult(f"item[{idx}]_fields", False, error=str(e)))

    # ── Check 4: Prompt template ──
    try:
        tpl = task.get_prompt_template()
        assert tpl.name, "Empty template name"
        assert tpl.template, "Empty template text"
        assert tpl.variables, "No template variables detected"
        assert tpl.answer_prefix is not None, "answer_prefix is None"

        # For multilingual tasks, check localized prefix
        if task_name == "mgsm" and lang != "en":
            expected_prefix = ANSWER_PREFIXES.get(lang, "Answer")
            assert tpl.answer_prefix == expected_prefix, (
                f"Wrong prefix: got {tpl.answer_prefix!r}, want {expected_prefix!r}"
            )

        report.checks.append(
            CheckResult(
                "prompt_template",
                True,
                f"name={tpl.name} vars={tpl.variables} prefix={tpl.answer_prefix!r} lang={tpl.requested_language}",
            )
        )
    except Exception as e:
        report.checks.append(CheckResult("prompt_template", False, error=str(e)))

    # ── Check 5: Prompt rendering ──
    for idx, item in enumerate(items):
        try:
            rendered = task.render_prompt(item)
            assert len(rendered) > 10, f"Rendered prompt too short: {len(rendered)} chars"

            # Check no unresolved {placeholders} from the template variables
            tpl = task.get_prompt_template_for_item(item)
            unresolved = [v for v in re.findall(r"\{(\w+)\}", rendered) if v in tpl.variables]
            assert not unresolved, f"Unresolved vars: {unresolved}"

            # For multilingual, verify localized prefix appears
            if task_name == "mgsm":
                expected_prefix = ANSWER_PREFIXES.get(lang, "Answer")
                assert expected_prefix in rendered, f"Rendered prompt missing prefix {expected_prefix!r}"

            report.checks.append(
                CheckResult(f"render[{idx}]", True, f"len={len(rendered)} first_80={trunc(rendered, 80)}")
            )
        except Exception as e:
            report.checks.append(CheckResult(f"render[{idx}]", False, error=str(e)))

    # ── Check 6: Semantic meta + parquet meta ──
    try:
        item = items[0]
        sem = item.semantic_meta()
        pq = item.parquet_meta()
        task_meta = task.get_task_metadata(item)
        prompt_meta = task.get_prompt_template_metadata(item)

        assert "task_name" in task_meta
        assert "task_source" in task_meta
        assert "prompt_template_name" in prompt_meta
        assert "prompt_template_hash" in prompt_meta
        assert isinstance(pq, dict)

        report.checks.append(
            CheckResult(
                "metadata_structure",
                True,
                f"semantic_keys={list(sem.keys())[:5]} task_meta_keys={list(task_meta.keys())}",
            )
        )
    except Exception as e:
        report.checks.append(CheckResult("metadata_structure", False, error=str(e)))

    return report


# ── Phase 2: Model-level checks ─────────────────────────────────────────


def check_model_pipeline(
    task_name: str,
    extra_kwargs: Dict[str, Any],
    model_runner,
    max_new_tokens: int,
) -> TaskReport:
    """Verify tokenization, generation, hidden states, and offsets with real model."""
    from openact_collect.tasks.registry import TaskRegistry
    from openact_collect.schema import CaptureSpec, GenerationSpec
    from openact_collect.extractors.hidden_state_extractor import HiddenStateExtractor
    from openact_collect.engine.offset_calculator import OffsetCalculator

    lang = extra_kwargs.get("language", "en")
    config_id = f"{task_name}_{lang}" if "language" in extra_kwargs else task_name
    report = TaskReport(config_id=config_id, task_name=task_name, language=lang)

    task = TaskRegistry.create(task_name, max_samples=MAX_SAMPLES_PER_TASK, **extra_kwargs)
    items = list(task.iter_items())

    capture_spec = CaptureSpec(
        hidden_states=True,
        hidden_states_layers=None,  # all layers
        hidden_states_dtype="float16",
        save_per_token=True,
        save_mean_states=True,
        save_prompt_last=True,
        compute_online_metrics=True,
    )
    gen_spec = GenerationSpec(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        seed=42,
    )

    extractor = HiddenStateExtractor(capture_spec, model_runner)
    offset_calc = OffsetCalculator(
        model_runner.tokenizer,
        special_ids=model_runner.get_special_ids(),
    )

    n_layers = model_runner.probed_n_layers
    hidden_dim = model_runner.get_model_spec().hidden_dim

    for idx, item in enumerate(items):
        rendered = task.render_prompt(item)
        messages = [{"role": "user", "content": rendered}]

        # ── Check: Chat template tokenization ──
        try:
            input_ids = model_runner.apply_chat_template(messages)
            n_input = int(input_ids.shape[-1])
            assert n_input > 0, "Zero input tokens"
            assert n_input < 4096, f"Suspiciously long: {n_input} tokens"

            # Verify round-trip: decode input_ids back and check it contains the question
            chat_text = model_runner.render_chat_text(messages)
            assert len(chat_text) > len(rendered), (
                "Chat text should be longer than raw prompt (has special tokens)"
            )

            report.checks.append(
                CheckResult(
                    f"tokenize[{idx}]",
                    True,
                    f"input_tokens={n_input} chat_text_len={len(chat_text)}",
                )
            )
        except Exception as e:
            report.checks.append(CheckResult(f"tokenize[{idx}]", False, error=str(e)))
            continue

        # ── Check: Generation ──
        try:
            gen_result = model_runner.generate(
                input_ids=input_ids,
                gen_spec=gen_spec,
                capture_hidden_states=True,
                capture_spec=capture_spec,
            )
            response_text = model_runner.decode(gen_result.token_ids, skip_special_tokens=True)
            n_gen = len(gen_result.token_ids)

            assert n_gen > 0, "Generated zero tokens"
            assert gen_result.input_length == n_input, (
                f"input_length mismatch: {gen_result.input_length} != {n_input}"
            )
            assert gen_result.generated_length == n_gen or gen_result.generated_length > 0
            assert gen_result.finish_reason in ("eos", "length", "other", "empty"), (
                f"Unknown finish_reason: {gen_result.finish_reason}"
            )
            assert len(response_text) > 0, "Empty response text"

            report.checks.append(
                CheckResult(
                    f"generate[{idx}]",
                    True,
                    f"gen_tokens={n_gen} finish={gen_result.finish_reason} "
                    f"response={trunc(response_text, 60)}",
                )
            )
        except Exception as e:
            report.checks.append(CheckResult(f"generate[{idx}]", False, error=str(e)))
            continue

        # ── Check: Hidden states extraction ──
        try:
            hs_data = extractor.extract(gen_result, input_ids=input_ids)
            assert hs_data is not None, "HiddenStateData is None"

            # Per-token states
            if hs_data.per_token_states is not None:
                pt = hs_data.per_token_states
                T, L, H = pt.shape
                assert T > 0, "per_token T=0"
                assert T == hs_data.n_tokens or T == n_gen, (
                    f"Token count mismatch: pt.T={T}, n_tokens={hs_data.n_tokens}, gen={n_gen}"
                )
                assert L > 0, f"per_token L=0 (expected ~{n_layers})"
                assert H == hidden_dim, f"per_token H={H} != hidden_dim={hidden_dim}"
                assert not np.all(pt == 0), "All per-token hidden states are zero"

                # Check dtype
                expected_dt = np.float16 if capture_spec.hidden_states_dtype == "float16" else np.float32
                assert pt.dtype == expected_dt, f"HS dtype={pt.dtype}, expected {expected_dt}"

                # Check no NaN/Inf
                pt_f32 = pt.astype(np.float32)
                assert np.all(np.isfinite(pt_f32)), "Hidden states contain NaN/Inf"

                report.checks.append(
                    CheckResult(
                        f"hs_per_token[{idx}]",
                        True,
                        f"shape=({T},{L},{H}) dtype={pt.dtype} "
                        f"norm_range=[{np.linalg.norm(pt_f32[0], axis=-1).min():.2f}, "
                        f"{np.linalg.norm(pt_f32[0], axis=-1).max():.2f}]",
                    )
                )
            else:
                report.checks.append(
                    CheckResult(
                        f"hs_per_token[{idx}]",
                        False,
                        error="per_token_states is None despite capture_spec.save_per_token=True",
                    )
                )

            # Mean states
            if hs_data.mean_states is not None:
                ms = hs_data.mean_states
                assert ms.ndim == 2, f"mean_states ndim={ms.ndim}"
                assert ms.shape[0] > 0, "mean_states L=0"
                assert ms.shape[1] == hidden_dim, f"mean_states H={ms.shape[1]}"
                assert not np.all(ms == 0), "Mean states all zero"
                assert np.all(np.isfinite(ms)), "Mean states contain NaN/Inf"
                report.checks.append(
                    CheckResult(f"hs_mean[{idx}]", True, f"shape={ms.shape} dtype={ms.dtype}")
                )
            else:
                report.checks.append(
                    CheckResult(
                        f"hs_mean[{idx}]",
                        False,
                        error="mean_states is None despite save_mean_states=True",
                    )
                )

            # Prompt-last states
            if hs_data.prompt_last_states is not None:
                pl = hs_data.prompt_last_states
                assert pl.ndim == 2, f"prompt_last ndim={pl.ndim}"
                assert pl.shape[1] == hidden_dim
                assert not np.all(pl == 0), "Prompt-last states all zero"
                report.checks.append(
                    CheckResult(f"hs_prompt_last[{idx}]", True, f"shape={pl.shape}")
                )
            else:
                # prompt_last can fail if no prompt step in hidden_states and input_ids absent
                report.checks.append(
                    CheckResult(
                        f"hs_prompt_last[{idx}]",
                        False,
                        error="prompt_last_states is None",
                    )
                )

            # Consistency: mean should approximate mean of per_token
            if hs_data.per_token_states is not None and hs_data.mean_states is not None:
                computed_mean = hs_data.per_token_states.astype(np.float32).mean(axis=0)
                diff = float(np.abs(computed_mean - hs_data.mean_states).max())
                ok = diff < 0.1  # float16 accumulation can have some error; plain bool for JSON
                report.checks.append(
                    CheckResult(
                        f"hs_mean_consistency[{idx}]",
                        ok,
                        f"max_diff={diff:.6f}" + ("" if ok else " (EXCEEDS 0.1 THRESHOLD)"),
                    )
                )

        except Exception as e:
            report.checks.append(CheckResult(f"hs_extract[{idx}]", False, error=str(e)))
            continue

        # ── Check: Token offset calculation ──
        try:
            offsets = offset_calc.compute_offsets(response_text, gen_result.token_ids)
            assert offsets.shape == (n_gen, 2), (
                f"Offsets shape={offsets.shape}, expected ({n_gen}, 2)"
            )
            assert offsets.dtype == np.int32, f"Offsets dtype={offsets.dtype}"

            # Monotonicity: starts should be non-decreasing
            starts = offsets[:, 0]
            for i in range(1, len(starts)):
                assert starts[i] >= starts[i - 1], (
                    f"Non-monotonic offset starts at token {i}: {starts[i]} < {starts[i - 1]}"
                )

            # start <= end for each token
            for i in range(len(offsets)):
                assert offsets[i, 0] <= offsets[i, 1], (
                    f"Token {i}: start={offsets[i, 0]} > end={offsets[i, 1]}"
                )

            # End should not exceed text length
            if n_gen > 0 and len(response_text) > 0:
                assert offsets[-1, 1] <= len(response_text), (
                    f"Offsets exceed text: {offsets[-1, 1]} > {len(response_text)}"
                )

            # Coverage: last token end should be near text length
            if n_gen > 0 and len(response_text) > 0:
                coverage = offsets[-1, 1] / len(response_text)
            else:
                coverage = 0.0

            report.checks.append(
                CheckResult(
                    f"offsets[{idx}]",
                    True,
                    f"shape={offsets.shape} coverage={coverage:.1%} "
                    f"range=[{offsets[0, 0]}, {offsets[-1, 1]}] text_len={len(response_text)}",
                )
            )

            # Validate through the offset calculator's own validation
            is_valid, issues = offset_calc.validate_offsets(
                response_text, gen_result.token_ids, offsets
            )
            if not is_valid:
                report.checks.append(
                    CheckResult(
                        f"offset_validation[{idx}]",
                        False,
                        error=f"Validation issues: {issues[:3]}",
                    )
                )
            else:
                report.checks.append(
                    CheckResult(f"offset_validation[{idx}]", True, "All offset checks passed")
                )

        except Exception as e:
            report.checks.append(CheckResult(f"offsets[{idx}]", False, error=str(e)))

        # Force cleanup between samples
        del gen_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


# ── Phase 3: Full CollectionRunner integration test ──────────────────────


def check_full_collection(
    task_name: str,
    extra_kwargs: Dict[str, Any],
    model_runner,
    max_new_tokens: int,
) -> TaskReport:
    """Run the full CollectionRunner and verify output files."""
    from openact_collect.engine.collector import CollectionRunner
    from openact_collect.schema import CaptureSpec, GenerationSpec
    from openact_collect.tasks.registry import TaskRegistry

    lang = extra_kwargs.get("language", "en")
    config_id = f"{task_name}_{lang}" if "language" in extra_kwargs else task_name
    report = TaskReport(config_id=config_id, task_name=task_name, language=lang)

    # Use temp directory for output
    tmpdir = Path(tempfile.mkdtemp(prefix=f"openact_sanity_{config_id}_"))

    try:
        task = TaskRegistry.create(task_name, max_samples=2, **extra_kwargs)

        capture_spec = CaptureSpec(
            hidden_states=True,
            save_per_token=True,
            save_mean_states=True,
            save_prompt_last=True,
            compute_online_metrics=True,
        )
        gen_spec = GenerationSpec(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            seed=42,
        )

        runner = CollectionRunner(
            model_manager=model_runner,
            task=task,
            output_dir=tmpdir,
            capture_spec=capture_spec,
            generation_spec=gen_spec,
        )

        stats = runner.run()

        # ── Check: Collection stats ──
        try:
            assert stats["processed"] > 0, "Zero samples processed"
            assert stats["ok"] > 0, "Zero OK samples"
            assert stats["duration_seconds"] > 0, "Zero duration"
            report.checks.append(
                CheckResult(
                    "collection_stats",
                    True,
                    f"processed={stats['processed']} ok={stats['ok']} "
                    f"errors={stats['error'] + stats['timeout']} "
                    f"duration={stats['duration_seconds']:.1f}s",
                )
            )
        except Exception as e:
            report.checks.append(CheckResult("collection_stats", False, error=str(e)))

        # ── Check: Output files exist ──
        try:
            assert (tmpdir / "manifest.json").exists(), "Missing manifest.json"
            assert (tmpdir / "data.parquet").exists(), "Missing data.parquet"
            assert (tmpdir / "tensors.zarr").exists(), "Missing tensors.zarr"
            assert (tmpdir / "_SUCCESS").exists(), "Missing _SUCCESS marker"
            report.checks.append(CheckResult("output_files", True, "All required files present"))
        except Exception as e:
            report.checks.append(CheckResult("output_files", False, error=str(e)))

        # ── Check: Manifest content ──
        try:
            manifest = json.loads((tmpdir / "manifest.json").read_text())
            assert manifest["model"]["name"], "Empty model name in manifest"
            assert manifest["dataset"]["name"] == task_name
            assert manifest["prompt"]["template_name"], "Empty template name"
            assert manifest["stats"]["n_samples_ok"] > 0
            report.checks.append(
                CheckResult(
                    "manifest_content",
                    True,
                    f"model={manifest['model']['name']} dataset={manifest['dataset']['name']}",
                )
            )
        except Exception as e:
            report.checks.append(CheckResult("manifest_content", False, error=str(e)))

        # ── Check: Load as Run and verify data ──
        try:
            from openact_core import Run

            run = Run(tmpdir)
            assert len(run) > 0, "Run has zero samples"
            assert run.n_valid > 0, "Run has zero valid samples"

            # Validate structural integrity
            issues = run.validate()
            assert not issues, f"Validation issues: {issues}"

            report.checks.append(
                CheckResult("run_load", True, f"n_total={len(run)} n_valid={run.n_valid}")
            )
        except Exception as e:
            report.checks.append(CheckResult("run_load", False, error=str(e)))
            return report

        # ── Check: Per-sample data in Run ──
        try:
            for sample in run.iter_valid():
                # Response
                assert sample.response_text, f"Empty response for sample {sample.sample_idx}"
                assert sample.n_tokens > 0, f"Zero tokens for sample {sample.sample_idx}"

                # Hidden states
                hs = sample.hidden_states
                assert hs.size > 0, f"Empty hidden states for sample {sample.sample_idx}"
                T, L, H = hs.shape
                assert T == sample.n_tokens, f"HS tokens {T} != sample tokens {sample.n_tokens}"
                assert not np.all(hs == 0), "All hidden states zero"
                assert np.all(np.isfinite(hs.astype(np.float32))), "NaN/Inf in HS"

                # Mean hidden states
                mean_hs = sample.mean_hidden_states
                assert mean_hs.size > 0, "Empty mean hidden states"
                assert mean_hs.ndim == 2

                # Token offsets (alignment)
                offsets = sample.token_offsets
                assert offsets.shape == (sample.n_tokens, 2)

                # Aligner
                aligner = sample.aligner
                assert aligner.n_tokens == sample.n_tokens
                val_issues = aligner.validate()

                # Metadata
                assert sample.prompt_text, "Empty prompt_text in meta"

                report.checks.append(
                    CheckResult(
                        f"run_sample[{sample.sample_idx}]",
                        True,
                        f"tokens={sample.n_tokens} hs={hs.shape} "
                        f"response={trunc(sample.response_text, 50)} "
                        f"alignment_issues={len(val_issues)}",
                    )
                )
                break  # Just check first valid sample in detail

        except Exception as e:
            report.checks.append(CheckResult("run_samples", False, error=str(e)))

        # ── Check: Parquet data columns ──
        try:
            import pandas as pd

            df = pd.read_parquet(tmpdir / "data.parquet")
            required_cols = [
                "sample_idx",
                "status",
                "prompt_text",
                "response_text",
                "ground_truth",
                "finish_reason",
                "n_prompt_tokens",
                "n_response_tokens",
                "task_name",
                "task_source",
                "prompt_template_name",
                "prompt_template_hash",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            assert not missing, f"Missing parquet columns: {missing}"

            # Check data integrity
            ok_rows = df[df["status"] == 1]
            assert len(ok_rows) > 0, "No OK rows in parquet"
            for _, row in ok_rows.iterrows():
                assert row["prompt_text"], "Empty prompt_text in parquet"
                assert row["response_text"], "Empty response_text in parquet"
                assert row["n_response_tokens"] > 0, "Zero response tokens in parquet"

            report.checks.append(
                CheckResult(
                    "parquet_integrity",
                    True,
                    f"rows={len(df)} ok_rows={len(ok_rows)} cols={len(df.columns)}",
                )
            )
        except Exception as e:
            report.checks.append(CheckResult("parquet_integrity", False, error=str(e)))

        # ── Check: Zarr tensor store ──
        try:
            import zarr

            z = zarr.open_group(str(tmpdir / "tensors.zarr"), mode="r")

            assert "tokens/ids" in z, "Missing tokens/ids"
            assert "tokens/offsets" in z, "Missing tokens/offsets"
            assert "tokens/sample_ptr" in z, "Missing tokens/sample_ptr"
            assert "hidden_states/per_token" in z, "Missing hidden_states/per_token"
            assert "hidden_states/mean" in z, "Missing hidden_states/mean"

            ptr = np.asarray(z["tokens/sample_ptr"][:])
            total_tokens = int(ptr[-1])
            hs_pt = z["hidden_states/per_token"]
            assert hs_pt.shape[0] == total_tokens, (
                f"HS tokens {hs_pt.shape[0]} != ptr total {total_tokens}"
            )

            report.checks.append(
                CheckResult(
                    "zarr_integrity",
                    True,
                    f"total_tokens={total_tokens} hs_shape={hs_pt.shape} "
                    f"ptr_len={len(ptr)}",
                )
            )
        except Exception as e:
            report.checks.append(CheckResult("zarr_integrity", False, error=str(e)))

        # ── Check: Bulk hidden state loading ──
        try:
            for reduction in ("mean", "first", "last"):
                hs_all = run.get_all_hidden_states(reduction=reduction, only_valid=True)
                assert hs_all.shape[0] == run.n_valid, (
                    f"{reduction}: shape[0]={hs_all.shape[0]} != n_valid={run.n_valid}"
                )
                assert hs_all.ndim == 3, f"{reduction}: ndim={hs_all.ndim}"
                assert not np.all(hs_all == 0), f"{reduction}: all zero"

            report.checks.append(
                CheckResult("bulk_hs_loading", True, f"All 3 reductions OK, shape={hs_all.shape}")
            )
        except Exception as e:
            report.checks.append(CheckResult("bulk_hs_loading", False, error=str(e)))

        # ── Check: Generation metrics in parquet ──
        try:
            metrics_df = run.get_all_generation_metrics(only_valid=True)
            if "perplexity" in metrics_df.columns:
                valid_ppl = metrics_df["perplexity"].dropna()
                assert len(valid_ppl) > 0, "No perplexity values"
                assert all(valid_ppl > 0), "Negative perplexity"
                report.checks.append(
                    CheckResult(
                        "generation_metrics",
                        True,
                        f"perplexity_range=[{valid_ppl.min():.2f}, {valid_ppl.max():.2f}]",
                    )
                )
            else:
                report.checks.append(
                    CheckResult(
                        "generation_metrics",
                        True,
                        "Metrics columns not present (OK if compute_online_metrics was off)",
                    )
                )
        except Exception as e:
            report.checks.append(CheckResult("generation_metrics", False, error=str(e)))

    except Exception as e:
        report.checks.append(
            CheckResult("collection_run", False, error=traceback.format_exc()[-500:])
        )
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    return report


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Final sanity check: full pipeline with real model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--task", default=None, help="Run only this task (e.g. gsm8k)")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Run only these tasks (e.g. math mgsm theoremqa). Overrides --task.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Run only this language for multilingual tasks",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max new tokens per generation (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip Phase 3 (full CollectionRunner test)",
    )
    parser.add_argument(
        "--collection-only",
        default=None,
        help="Run Phase 3 only for this task (e.g. gsm8k)",
    )
    parser.add_argument("--json", default=None, metavar="FILE", help="Write JSON report to file (optional)")
    args = parser.parse_args()

    # Filter configs
    configs = list(CAPABILITY_CONFIGS)
    if args.tasks:
        allow = {name.strip() for name in args.tasks}
        configs = [(t, e) for t, e in configs if t in allow]
        if args.language:
            configs = [
                (t, e)
                for t, e in configs
                if e.get("language") == args.language
                or (not e.get("language") and args.language == "en")
            ]
    elif args.task:
        configs = [(t, e) for t, e in configs if t == args.task]
        if args.language:
            configs = [
                (t, e)
                for t, e in configs
                if e.get("language") == args.language
                or (not e.get("language") and args.language == "en")
            ]
    if not configs:
        print(
            f"No matching configs for task={args.task}, tasks={args.tasks}, language={args.language}"
        )
        sys.exit(1)

    all_reports: List[TaskReport] = []
    start_time = time.time()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: Dataset + Prompt (no model)
    # ════════════════════════════════════════════════════════════════════
    section("PHASE 1: Dataset Download + Prompt Rendering")
    print(f"  Testing {len(configs)} task configs, {MAX_SAMPLES_PER_TASK} samples each\n")

    for task_name, extra in configs:
        label = f"{task_name}_{extra.get('language', '')}" if extra.get("language") else task_name
        try:
            report = check_dataset_and_prompts(task_name, dict(extra))
            all_reports.append(report)
            status = "PASS" if report.all_passed else "FAIL"
            icon = "✅" if report.all_passed else "❌"
            print(f"  {icon} {label:25s}  {report.passed}/{report.passed + report.failed} checks")
            for c in report.checks:
                if not c.passed:
                    print(f"       ❌ {c.name}: {c.error}")
        except Exception as e:
            print(f"  ❌ {label:25s}  EXCEPTION: {e}")
            r = TaskReport(
                config_id=label,
                task_name=task_name,
                language=extra.get("language", "en"),
            )
            r.checks.append(CheckResult("phase1", False, error=str(e)))
            all_reports.append(r)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: Model Pipeline (requires GPU)
    # ════════════════════════════════════════════════════════════════════
    section("PHASE 2: Model Generation + Hidden State Capture")
    print(f"  Model: {args.model}")
    print(f"  Max new tokens: {args.max_tokens}")
    print("  Loading model...\n")

    from openact_collect.engine.model_manager import ModelRunner

    model_runner = ModelRunner(
        model_name_or_path=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model_runner.load()

    spec = model_runner.get_model_spec()
    print(f"  Model loaded: {spec.name}")
    print(f"  Architecture: {spec.architecture}")
    print(f"  Layers: {spec.n_layers} (probed: {model_runner.probed_n_layers})")
    print(f"  Decoder layers: {spec.n_decoder_layers} (probed: {model_runner.decoder_n_layers})")
    print(f"  Hidden dim: {spec.hidden_dim}")
    print(f"  Vocab size: {spec.vocab_size}")
    print(f"  Device: {model_runner.device}")
    print()

    for task_name, extra in configs:
        label = f"{task_name}_{extra.get('language', '')}" if extra.get("language") else task_name
        try:
            report = check_model_pipeline(task_name, dict(extra), model_runner, args.max_tokens)
            all_reports.append(report)
            icon = "✅" if report.all_passed else "❌"
            print(f"  {icon} {label:25s}  {report.passed}/{report.passed + report.failed} checks")
            for c in report.checks:
                if not c.passed:
                    print(f"       ❌ {c.name}: {c.error}")
                elif "hs_per_token" in c.name or "generate" in c.name:
                    # Print key details even for passing checks
                    print(f"       ℹ️  {c.name}: {c.details}")
        except Exception as e:
            print(f"  ❌ {label:25s}  EXCEPTION: {e}")
            traceback.print_exc(limit=3)
            r = TaskReport(
                config_id=label,
                task_name=task_name,
                language=extra.get("language", "en"),
            )
            r.checks.append(CheckResult("phase2", False, error=str(e)))
            all_reports.append(r)

        # Memory cleanup between tasks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3: Full CollectionRunner (1-2 tasks to keep it manageable)
    # ════════════════════════════════════════════════════════════════════
    if not args.skip_collection:
        section("PHASE 3: Full CollectionRunner Integration")

        # Select tasks for full collection test
        if args.collection_only:
            collection_configs = [(t, e) for t, e in configs if t == args.collection_only]
        else:
            # Default: gsm8k + mmlu (en) when present; else first configs in current filter
            collection_configs = [
                c
                for c in configs
                if c[0] in ("gsm8k", "mmlu") and c[1].get("language", "en") == "en"
            ][:2]
            ml = [c for c in configs if c[0] == "mgsm" and c[1].get("language") == "zh"]
            if ml and ml[0] not in collection_configs:
                collection_configs.append(ml[0])
            if not collection_configs:
                collection_configs = configs[: min(2, len(configs))]

        print(f"  Running full collection for {len(collection_configs)} configs\n")

        for task_name, extra in collection_configs:
            label = f"{task_name}_{extra.get('language', '')}" if extra.get("language") else task_name
            try:
                report = check_full_collection(
                    task_name, dict(extra), model_runner, args.max_tokens
                )
                all_reports.append(report)
                icon = "✅" if report.all_passed else "❌"
                print(f"  {icon} {label:25s}  {report.passed}/{report.passed + report.failed} checks")
                for c in report.checks:
                    if not c.passed:
                        print(f"       ❌ {c.name}: {c.error}")
                    elif c.name in ("collection_stats", "zarr_integrity", "parquet_integrity"):
                        print(f"       ℹ️  {c.name}: {c.details}")
            except Exception as e:
                print(f"  ❌ {label:25s}  EXCEPTION: {e}")
                traceback.print_exc(limit=3)
                r = TaskReport(
                    config_id=label,
                    task_name=task_name,
                    language=extra.get("language", "en"),
                )
                r.checks.append(CheckResult("phase3", False, error=str(e)))
                all_reports.append(r)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════
    section("FINAL SUMMARY")

    total_checks = sum(r.passed + r.failed for r in all_reports)
    total_passed = sum(r.passed for r in all_reports)
    total_failed = sum(r.failed for r in all_reports)
    elapsed = time.time() - start_time

    # Group by phase
    phase_reports = {
        "Phase 1 (Data+Prompt)": [],
        "Phase 2 (Model)": [],
        "Phase 3 (Collection)": [],
    }
    n_configs = len(configs)
    for i, r in enumerate(all_reports):
        if i < n_configs:
            phase_reports["Phase 1 (Data+Prompt)"].append(r)
        elif i < 2 * n_configs:
            phase_reports["Phase 2 (Model)"].append(r)
        else:
            phase_reports["Phase 3 (Collection)"].append(r)

    for phase_name, reports in phase_reports.items():
        if not reports:
            continue
        p = sum(r.passed for r in reports)
        f = sum(r.failed for r in reports)
        tasks_ok = sum(1 for r in reports if r.all_passed)
        icon = "✅" if f == 0 else "❌"
        print(
            f"  {icon} {phase_name:30s}  {tasks_ok}/{len(reports)} tasks OK  "
            f"({p} checks passed, {f} failed)"
        )

    print(f"\n  {'─' * 60}")
    print(f"  Total checks:   {total_checks}")
    print(f"  Passed:         {total_passed}")
    print(f"  Failed:         {total_failed}")
    print(f"  Duration:       {elapsed:.1f}s")
    print(f"  Model:          {args.model}")
    print(f"  {'─' * 60}")

    if args.json:
        def _json_default(obj: Any) -> Any:
            # NumPy scalars (e.g. np.bool_, np.float32) are not json-serializable; NumPy 2 may
            # report some as type name "bool".
            try:
                if isinstance(obj, np.generic):
                    return obj.item()
            except (ImportError, AttributeError, ValueError, TypeError):
                pass
            raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serializable")

        out = {
            "model": args.model,
            "duration_s": float(elapsed),
            "reports": [
                {
                    "config_id": r.config_id,
                    "task": r.task_name,
                    "language": r.language,
                    "all_passed": bool(r.all_passed),
                    "checks": [
                        {
                            "name": c.name,
                            "passed": bool(c.passed),
                            "details": c.details if isinstance(c.details, str) else str(c.details),
                            "error": None if c.error is None else str(c.error),
                        }
                        for c in r.checks
                    ],
                }
                for r in all_reports
            ],
        }
        Path(args.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        print(f"\n  Wrote JSON report to {args.json}")

    if total_failed > 0:
        print(f"\n  ⚠️  {total_failed} CHECK(S) FAILED — review output above")
        print("\n  Failed checks:")
        for r in all_reports:
            for c in r.checks:
                if not c.passed:
                    print(f"    [{r.config_id}] {c.name}: {c.error}")
        sys.exit(1)
    else:
        print(f"\n  🎉 ALL {total_checks} CHECKS PASSED — pipeline verified!")
        sys.exit(0)


if __name__ == "__main__":
    main()
