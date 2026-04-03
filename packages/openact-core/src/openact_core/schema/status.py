from enum import IntEnum
class SampleStatus(IntEnum):
    UNPROCESSED = 0
    OK = 1
    SKIPPED = 2
    TIMEOUT = 3
    ERROR = 4
    INVALID_INPUT = 5
    EMPTY_RESPONSE = 6
    @classmethod
    def from_string(cls, s: str) -> "SampleStatus":
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
        return self == SampleStatus.OK
    def is_failure(self) -> bool:
        return self in (
            SampleStatus.TIMEOUT,
            SampleStatus.ERROR,
            SampleStatus.INVALID_INPUT,
            SampleStatus.EMPTY_RESPONSE,
        )
    def is_terminal(self) -> bool:
        return self != SampleStatus.UNPROCESSED
