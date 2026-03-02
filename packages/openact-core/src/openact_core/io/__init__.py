from openact_core.io.run import Run
from openact_core.io.sample import Sample
from openact_core.io.loader import DataLoader, LoadedData

__all__ = ['Run', 'Sample', 'DataLoader', 'LoadedData']

# backward compat
try:
    from openact_core.io.loader import AlignedLoader, AlignedDataset
    __all__.extend(['AlignedLoader', 'AlignedDataset'])
except ImportError:
    pass

try:
    from openact_core.io.batch_reader import HiddenStateLoader, RunSnapshot
    __all__.extend(['HiddenStateLoader', 'RunSnapshot'])
except ImportError:
    pass