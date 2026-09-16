"""Official WildJailbreak training set, strictly vanilla_harmful prompts only."""
from openact_collect.data import HFDatasetSpec
from openact_collect.schema import GenerationProfile
from openact_collect.tasks.registry import TaskRegistry
from openact_collect.tasks.safety.base import SafetyTask


@TaskRegistry.register('wildjailbreak')
class WildJailbreakTask(SafetyTask):
    source = 'allenai/wildjailbreak'
    language = 'en'
    default_template = 'zot'

    def __init__(self, max_samples=None, split='harmful', template='zot', profiles=None, **kwargs):
        if split not in ('harmful', 'all', '*'):
            raise ValueError('This adapter selects only WildJailbreak vanilla_harmful')
        kwargs.pop('include_artifacts', None)
        super().__init__(max_samples=max_samples, split=split, template=template,
                         profiles=profiles or {'greedy': GenerationProfile(name='greedy')},
                         include_artifacts=False, **kwargs)
        self._behaviors = None

    def load_behaviors(self):
        if self._behaviors is None:
            dataset = self.load_hf_dataset(HFDatasetSpec(
                name=self.source, config='train', split='train',
                loader_kwargs={'delimiter': '\t', 'keep_default_na': False},
            ))
            selected = []
            for source_idx, row in enumerate(dataset):
                if row['data_type'] != 'vanilla_harmful':
                    continue
                prompt = row['vanilla']
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f'Empty vanilla_harmful prompt at source row {source_idx}')
                selected.append({'behavior_id': f'vanilla_harmful_{source_idx}',
                                 'goal': prompt, 'category': 'vanilla_harmful',
                                 'split': 'harmful', 'source': self.source})
            if not selected:
                raise ValueError('No vanilla_harmful rows in the selected WildJailbreak source')
            self._behaviors = selected
        return self._behaviors
