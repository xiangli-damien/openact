"""Sample status enumeration for tracking collection state."""

from enum import IntEnum


class SampleStatus(IntEnum):
    """
    Status codes for sample processing state.
    
    These are stored as int8 in the zarr arrays and parquet files,
    enabling efficient filtering and resume logic.
    
    Attributes:
        UNPROCESSED: Sample has not been processed yet (default state)
        OK: Sample was processed successfully with valid activations
        SKIPPED: Sample was intentionally skipped (e.g., filtered out)
        TIMEOUT: Sample processing exceeded time limit
        ERROR: Sample processing failed with an exception
        INVALID_INPUT: Sample had invalid input data
        EMPTY_RESPONSE: Model generated empty response
    """
    
    UNPROCESSED = 0
    OK = 1
    SKIPPED = 2
    TIMEOUT = 3
    ERROR = 4
    INVALID_INPUT = 5
    EMPTY_RESPONSE = 6
    
    @classmethod
    def from_string(cls, s: str) -> "SampleStatus":
        """Convert string to SampleStatus."""
        mapping = {
            "unprocessed": cls.UNPROCESSED,
            "ok": cls.OK,
            "skipped": cls.SKIPPED,
            "timeout": cls.TIMEOUT,
            "error": cls.ERROR,
            "invalid_input": cls.INVALID_INPUT,
            "empty_response": cls.EMPTY_RESPONSE,
        }
        return mapping.get(s.lower(), cls.ERROR)
    
    def is_success(self) -> bool:
        """Check if this status represents successful processing."""
        return self == SampleStatus.OK
    
    def is_failure(self) -> bool:
        """Check if this status represents a failure."""
        return self in (
            SampleStatus.TIMEOUT,
            SampleStatus.ERROR,
            SampleStatus.INVALID_INPUT,
            SampleStatus.EMPTY_RESPONSE,
        )
    
    def is_terminal(self) -> bool:
        """Check if this status is terminal (won't change on retry)."""
        return self != SampleStatus.UNPROCESSED
