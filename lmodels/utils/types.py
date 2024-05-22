from enum import Enum

import torch


class DType(Enum):
    """
    An enumeration of the data types supported by PyTorch.
    This enum is used in config dataclasses due to the limitations of `hydra` regarding typing attributes with custom classes.
    """

    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"
    uint8 = "uint8"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"

    @property
    def torch(self) -> torch.dtype:
        """Returns the corresponding `torch.dtype` for the enum value."""

        return getattr(torch, self.name)

    def __str__(self) -> str:
        return self.value
