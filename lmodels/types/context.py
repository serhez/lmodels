import numpy as np
import numpy.typing as npt

from lmodels.protocols import Dataset

Message = str
"""A single simple message."""

AnnotatedMessage = dict[str, str]
"""
A single message with model-specific fields, such as `role` for OpenAI's models.
Must contain a `content` field.
"""

Conversation = list[Message]
"""A list of messages forming a conversation."""

AnnotatedConversation = list[AnnotatedMessage]
"""A list of messages with model-specific fields forming a conversation."""

Context = (
    Message
    | AnnotatedMessage
    | Conversation
    | AnnotatedConversation
    | list[Conversation]
    | list[AnnotatedConversation]
    # Alternative representations
    | npt.NDArray[
        np.str_
    ]  # can be equivalent to List[str] or List[List[str]], depending on the shape
    | Dataset  # equivalent to List[List[str]], where the length of each inner list is 1
)
"""
The possible types of context input for the `generate` method and its derivatives.
Note that a list of messages (`List[Message]`) is equivalent to a single conversation (`Conversation`), and the same applies to annotated messages.
"""
