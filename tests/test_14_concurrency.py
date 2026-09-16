from copy import deepcopy

from scripts.benchmark_collection_concurrency import assess


def cases():
    math = {'dataset': 'math', 'sample_ids': ['math_1'], 'token_ids': [[1, 2]]}
    mmlu = {'dataset': 'mmlu', 'sample_ids': ['mmlu_1'], 'token_ids': [[3, 4]]}
    return ({'wall_seconds': 100, 'workers': [math]},
            {'wall_seconds': 50, 'workers': [mmlu]},
            {'wall_seconds': 110, 'workers': [deepcopy(math), deepcopy(mmlu)],
             'peak_total_gpu_mib': 24000})


def test_parallel_gate_requires_real_combined_throughput_gain():
    math, mmlu, pair = cases()
    assert assess(math, mmlu, pair)['eligible']
    pair['wall_seconds'] = 155
    assert not assess(math, mmlu, pair)['eligible']


def test_parallel_gate_requires_identical_tokens_and_sample_ids():
    math, mmlu, pair = cases()
    pair['workers'][1]['token_ids'] = [[3, 5]]
    assert not assess(math, mmlu, pair)['eligible']
    math, mmlu, pair = cases()
    pair['workers'][0]['sample_ids'] = ['wrong_question']
    assert not assess(math, mmlu, pair)['eligible']


def test_parallel_gate_keeps_eight_gib_headroom():
    math, mmlu, pair = cases()
    pair['peak_total_gpu_mib'] = 35000
    assert not assess(math, mmlu, pair)['eligible']
