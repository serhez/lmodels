from .logging import NullLogger
from .types import DType
from .usage import Usage

__all__ = ["NullLogger", "DType", "Usage", "classproperty"]


# Damn you, Python developers...
class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)  # type: ignore[reportOptionalCall]
