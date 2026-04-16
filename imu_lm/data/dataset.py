"""Minimal base class expected by imuposerdataset.py."""


class Sequence:
    """Abstract base for sequential IMU datasets."""

    def get_length(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.get_length()
