"""Prepare existing WildJailbreak adversarial benchmark prompts, no attack generation."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import pandas as pd
from datasets import load_dataset
from openact_collect.tasks.safety.wildjailbreak import WildJailbreakTask

REVISION = '5ddc12a7894f842b0619b8e1c7ee496b198af009'


def benchmark_rows(source, seed=42):
    candidates = [i for i, row in enumerate(source) if row['data_type'] == 'adversarial_harmful']
    random.Random(seed).shuffle(candidates)
    seen_behaviors, seen_prompts, records = set(), set(), []
    for index in candidates:
        row = source[index]
        prompt = row['adversarial'].strip()
        vanilla = row['vanilla'].strip()
        if not prompt or not vanilla:
            raise ValueError(f'Empty benchmark prompt/underlying behavior at row {index}')
        behavior = hashlib.sha256(vanilla.encode()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        if behavior in seen_behaviors or prompt_hash in seen_prompts:
            continue
        seen_behaviors.add(behavior); seen_prompts.add(prompt_hash)
        records.append(dict(sample_idx=len(records),sample_id=f'wildjailbreak_adversarial_harmful_{index}',
            prompt_text=prompt+'\n\nPlease reason step by step.',ground_truth='harmful',language='en',
            behavior_id=behavior,source_row=index,category='adversarial_harmful',split='harmful',
            prompt_variant='official_adversarial',attack_method='dataset_provided',profile='greedy',rep_idx=0,
            source='allenai/wildjailbreak',task_name='wildjailbreak',task_type='safety',
            prompt_template_variant='zot',underlying_behavior_sha256=behavior))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Prepared dataset output must be a fresh directory')
    source=load_dataset('allenai/wildjailbreak','train',split='train',revision=REVISION,
                        delimiter='\t',keep_default_na=False)
    rows=benchmark_rows(source)
    args.output.mkdir(parents=True)
    for start in range(0,len(rows),5000):
        pd.DataFrame(rows[start:start+5000]).to_parquet(args.output/f'part_{start//5000:05d}.parquet',index=False)
    template=WildJailbreakTask().get_prompt_template().to_dict()
    manifest=dict(task='wildjailbreak',task_source='allenai/wildjailbreak',task_type='safety',split='harmful',
        language='en',max_samples=None,prompt_template_variant='zot',prompt_template=template,
        task_config={'subset':'adversarial_harmful','one_variant_per_underlying_behavior':True,'seed':42},
        dataset_sources=[dict(name='allenai/wildjailbreak',config='train',split='train',revision=REVISION,
            fingerprint=source._fingerprint,loader_kwargs={'delimiter':'\t','keep_default_na':False})],
        n_samples=len(rows),reference_completion_used=False)
    (args.output/'prepared_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'prepared_samples':len(rows),'source':'adversarial_harmful','one_per_behavior':True,'output':str(args.output)}))


if __name__=='__main__':
    main()
