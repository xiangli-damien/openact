"""
Aggregate statistics for a data-collection run.

:class:`RunStats` is a plain dataclass built by :attr:`Run.stats`.  It is
cached on first access so repeated reads (e.g. in a dashboard) are cheap.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class RunStats:
    """Snapshot of high-level run metrics.

    Attributes
    ----------
    n_samples_total : int
        Total number of samples in the run (all statuses).
    n_samples_ok : int
        Samples that completed successfully (``SampleStatus.OK``).
    n_samples_error : int
        Samples that failed (``ERROR`` + ``TIMEOUT``).
    n_tokens_total : int
        Sum of response token counts across all OK samples.
    duration_seconds : float or None
        Wall-clock time of the collection run.
    """

    n_samples_total: int = 0
    n_samples_ok: int = 0
    n_samples_error: int = 0
    n_tokens_total: int = 0
    duration_seconds: Optional[float] = None
    run_id: Optional[str] = None
    model: Optional[str] = None
    dataset: Optional[str] = None
    n_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    hidden_states_shape: Optional[tuple] = None
    is_complete: bool = False
    status_counts: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        fields = []
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None and value != 0 and value is not False:
                fields.append(f"{field_name}={value!r}")
        return f"RunStats({', '.join(fields)})"