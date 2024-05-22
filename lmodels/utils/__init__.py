from .context import merge_system_messages
from .logging import NullLogger
from .usage import Usage

__all__ = ["NullLogger", "Usage", "classproperty", "merge_system_messages"]


# Damn you, Python developers...
class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)  # type: ignore[reportOptionalCall]
