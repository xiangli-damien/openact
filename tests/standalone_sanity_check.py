import sys
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def _candidate_src_dirs(root: Path, pkg: str):
    return [
        root / pkg / 'src',
        root.parent / pkg / 'src',
        root / 'packages' / pkg / 'src',
        root.parent / 'packages' / pkg / 'src',
    ]

for pkg in ('openact-core', 'openact-collect', 'openact-eval'):
    for src in _candidate_src_dirs(ROOT, pkg):
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
            break
_pass, _fail, _skip = (0, 0, 0)
MGSM_LANGS = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
BELEBELE_LANGS = ['en', 'zh', 'ja', 'de', 'fr', 'es', 'ru', 'ar', 'hi', 'ko', 'pt', 'it', 'th', 'vi', 'bn', 'sw', 'te', 'tr']

def section(title):
    print(f"\n{'=' * 80}")
    print(f'  {title}')
    print(f"{'=' * 80}")

def check(name, fn):
    global _pass, _fail, _skip
    try:
        r = fn()
        if r == 'SKIP':
            _skip += 1
            print(f'  ⏭  {name}')
        else:
            _pass += 1
            print(f'  ✅ {name}')
    except Exception as e:
        _fail += 1
        print(f'  ❌ {name}: {e}')
        traceback.print_exc(limit=3)

def trunc(s, n=120):
    s = str(s) if s is not None else 'None'
    return s[:n] + ('…' if len(s) > n else '')
section('1. IMPORTS')

def c_imports():
    import openact_core
    import openact_collect
    import openact_eval
    eval_ver = getattr(openact_eval, '__version__', 'n/a')
    print(f'       core={openact_core.__version__} collect={openact_collect.__version__} eval={eval_ver}')
check('Import all packages', c_imports)

def c_registry():
    from openact_collect.tasks.registry import TaskRegistry
    tasks = TaskRegistry.list()
    print(f'       Tasks ({len(tasks)}): {tasks}')
    assert len(tasks) >= 10
check('Task registry', c_registry)
section('2. MONOLINGUAL DATASETS — download, fields, GT, prompt')
MONO = [('gsm8k', {}, 'numeric'), ('mmlu', {}, 'letter'), ('math', {}, 'any'), ('theoremqa', {}, 'any'), ('arc_challenge', {}, 'letter'), ('commonsenseqa', {}, 'letter'), ('truthfulqa', {}, 'any'), ('humaneval', {}, 'none'), ('ifeval', {}, 'none'), ('belebele', {'language': 'en'}, 'letter')]
for task_name, extra, gt_type in MONO:

    def make(tn=task_name, ex=extra, gtt=gt_type):

        def fn():
            from openact_collect.tasks.registry import TaskRegistry
            task = TaskRegistry.create(tn, max_samples=3, **ex)
            items = list(task.iter_items())
            assert items, 'No items'
            item = items[0]
            rendered = task.render_prompt(item)
            tpl = task.get_prompt_template()
            print(f'       source={task.source} split={task.split}')
            print(f'       template={tpl.name} vars={tpl.variables}')
            print(f'       GT={trunc(item.ground_truth, 60)}  fields={list(item.prompt_fields.keys())}')
            print(f'       Rendered: {trunc(rendered, 150)}')
            if gtt == 'numeric':
                assert item.ground_truth is not None
                float(str(item.ground_truth).replace(',', ''))
            elif gtt == 'letter':
                assert str(item.ground_truth).strip() in 'ABCDE'
            elif gtt == 'any':
                assert item.ground_truth is not None
            assert len(rendered) > 10
        return fn
    check(f'{task_name}', make())
section('3. MGSM — all 11 languages')
for lang in MGSM_LANGS:

    def make(l=lang):

        def fn():
            from openact_collect.tasks.registry import TaskRegistry
            from openact_core.tasks.templates import ANSWER_PREFIXES
            task = TaskRegistry.create('mgsm', max_samples=2, language=l)
            items = list(task.iter_items())
            assert items
            item = items[0]
            rendered = task.render_prompt(item)
            tpl = task.get_prompt_template()
            expected_prefix = ANSWER_PREFIXES.get(l, 'Answer')
            assert tpl.answer_prefix == expected_prefix
            assert item.ground_truth is not None
            float(str(item.ground_truth).replace(',', ''))
            assert expected_prefix in rendered
            print(f"       prefix={tpl.answer_prefix!r} GT={item.ground_truth} Q={trunc(item.prompt_fields.get('question', ''), 80)}")
        return fn
    check(f'mgsm_{lang}', make())
section('4. BELEBELE — all 18 languages')
for lang in BELEBELE_LANGS:

    def make(l=lang):

        def fn():
            from openact_collect.tasks.registry import TaskRegistry
            task = TaskRegistry.create('belebele', max_samples=2, language=l)
            items = list(task.iter_items())
            assert items
            item = items[0]
            rendered = task.render_prompt(item)
            assert item.ground_truth in ('A', 'B', 'C', 'D')
            expected = 'ABCD'[int(item.meta['correct_answer_num']) - 1]
            assert item.ground_truth == expected
            print(f"       GT={item.ground_truth} correct_num={item.meta['correct_answer_num']} passage={trunc(item.meta.get('passage', ''), 80)}")
        return fn
    check(f'belebele_{lang}', make())
section('5. SAFETY DATASETS')
for tn in ['jbb', 'advbench', 'xstest']:

    def make(t=tn):

        def fn():
            from openact_collect.tasks.registry import TaskRegistry
            from openact_collect.schema import DEFAULT_SAFETY_PROFILES
            profiles = {'greedy': DEFAULT_SAFETY_PROFILES['greedy']}
            kwargs = {'max_samples': 3, 'profiles': profiles}
            if t == 'jbb':
                kwargs['include_artifacts'] = False
            task = TaskRegistry.create(t, **kwargs)
            items = list(task.iter_items())
            assert items
            item = items[0]
            rendered = task.render_prompt(item)
            assert rendered == item.prompt_text
            print(f'       behaviors={len(items)} bid={item.behavior_id} split={item.split}')
            print(f'       prompt={trunc(item.prompt_text, 100)}')
        return fn
    check(f'{tn}', make())
section('6. PARSERS — synthetic responses')
PARSER_TESTS = [('gsm8k', 'Answer: 42', '42', '42', True), ('gsm8k', '#### 1234', '1234', '1234', True), ('gsm8k', 'Answer: 42', '42', '45', False), ('mgsm', '答案: 150', '150', '150', True), ('mmlu', 'Answer: C', 'C', 'C', True), ('mmlu', 'The answer is (B).', 'B', 'A', False), ('math', '$\\boxed{\\frac{1}{2}}$', '\\frac{1}{2}', '\\frac{1}{2}', True), ('math', '$\\boxed{42}$', '42', '42', True), ('arc_challenge', 'Answer: A', 'A', 'A', True), ('belebele', '(D)', 'D', 'D', True), ('commonsenseqa', 'Answer: B', 'B', 'B', True), ('jbb', 'I cannot help with that.', 'refusal', None, None), ('jbb', 'Sure! Here you go.', 'compliance', None, None)]
for task, resp, exp_ext, gt, should_match in PARSER_TESTS:

    def make(t=task, r=resp, e=exp_ext, g=gt, sm=should_match):

        def fn():
            from openact_core.tasks.parsers import get_parser
            from openact_eval.matchers import get_matcher
            p = get_parser(t)
            ext = p.extract(r)
            norm = p.normalize(ext)
            assert ext is not None
            if t in ('jbb', 'advbench', 'xstest'):
                assert ext == e
                print(f'       {r[:50]} → {ext}')
                return
            if sm is None:
                return
            m = get_matcher(t)
            gt_norm = p.normalize(g)
            result = m.match(norm, gt_norm)
            assert result == sm, f'match={result}, expected={sm}'
            print(f'       {r[:50]} → ext={ext!r} match={result}')
        return fn
    check(f'parse {task}: {exp_ext}', make())
section('7. END-TO-END SIMULATION (dataset → prompt → fake response → parse → match)')
E2E_TASKS = [('gsm8k', {}), ('mmlu', {}), ('math', {}), ('arc_challenge', {}), ('commonsenseqa', {}), ('belebele', {'language': 'en'}), ('theoremqa', {}), ('mgsm', {'language': 'en'}), ('mgsm', {'language': 'zh'}), ('mgsm', {'language': 'ja'}), ('belebele', {'language': 'zh'}), ('belebele', {'language': 'ar'})]
for tn, ex in E2E_TASKS:
    label = f"{tn}_{ex.get('language', '')}" if ex else tn

    def make(t=tn, e=ex, lb=label):

        def fn():
            from openact_collect.tasks.registry import TaskRegistry
            from openact_core.tasks.parsers import get_parser
            from openact_eval.matchers import get_matcher
            task = TaskRegistry.create(t, max_samples=2, **e)
            items = list(task.iter_items())
            assert items
            p = get_parser(t)
            m = get_matcher(t)
            item = items[0]
            gt = item.ground_truth
            if gt is None:
                print(f'       (no GT, skip matching)')
                return
            if t in ('gsm8k', 'mgsm'):
                fake = f'Step 1: ...\nAnswer: {gt}'
            elif t in ('mmlu', 'arc_challenge', 'commonsenseqa', 'belebele'):
                fake = f'The answer is ({gt}).\nAnswer: {gt}'
            elif t == 'math':
                fake = f'$\\boxed{{{gt}}}$'
            else:
                fake = f'Answer: {gt}'
            ext = p.extract(fake)
            norm = p.normalize(ext)
            gt_norm = p.normalize(gt)
            ok = m.match(norm, gt_norm)
            print(f'       GT={trunc(gt, 30)} ext={trunc(ext, 30)} match={ok}')
            assert ok, f'E2E fail: {norm!r} vs {gt_norm!r}'
        return fn
    check(f'e2e {label}', make())
section('8. DESCRIPTORS & TEMPLATES CONSISTENCY')

def c_descriptors():
    from openact_core.tasks.descriptor import TASK_DESCRIPTORS, EVAL_PARSER
    from openact_core.tasks.parsers import list_parsers
    from openact_collect.tasks.registry import TaskRegistry
    parsers = set(list_parsers())
    collect_tasks = set(TaskRegistry.list())
    for name, desc in sorted(TASK_DESCRIPTORS.items()):
        if desc.eval_strategy == EVAL_PARSER and name in collect_tasks:
            ok = desc.parser_type in parsers or name in parsers
            assert ok, f'Task {name} needs parser {desc.parser_type!r}'
    print(f'       {len(TASK_DESCRIPTORS)} descriptors checked (collect tasks only)')
check('Descriptor-parser consistency', c_descriptors)

def c_templates():
    from openact_core.tasks.templates import has_template
    tasks = ['gsm8k', 'mgsm', 'mmlu', 'math', 'theoremqa', 'arc_challenge', 'commonsenseqa', 'belebele', 'truthfulqa', 'humaneval', 'ifeval']
    for t in tasks:
        assert has_template(t), f'No template for {t}'
    print(f'       All {len(tasks)} tasks have templates')
check('Template registration', c_templates)
section('9. SMOKE RUN (if exists)')
SMOKE = ROOT / 'runs' / 'smoke'

def c_smoke():
    if not SMOKE.exists():
        return 'SKIP'
    from openact_core import Run
    import numpy as np
    run = Run(SMOKE)
    s = run[int(run.get_valid_indices()[0])]
    hs = s.hidden_states
    print(f'       valid={run.n_valid} hs_shape={hs.shape} tokens={s.n_tokens}')
    print(f'       response: {s.response_text[:100]}…')
    if hs.size > 0:
        assert hs.shape[0] == s.n_tokens
        assert not np.all(hs == 0)
check('Smoke run', c_smoke)
section('SUMMARY')
total = _pass + _fail + _skip
print(f'\n  ✅ Passed:  {_pass}')
print(f'  ❌ Failed:  {_fail}')
print(f'  ⏭  Skipped: {_skip}')
print(f'  Total:     {total}')
print()
if _fail > 0:
    print('  ⚠️  SOME CHECKS FAILED — fix before collecting!')
    sys.exit(1)
else:
    print('  🎉 All checks passed — safe to collect!')
    sys.exit(0)
