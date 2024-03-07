from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from ldata import Dataset  # TODO: Drop this dependency with own Dataset interface
from mloggers import Logger
from mloggers.progress import log_progress

Message = str
"""A single simple message."""

AnnotatedMessage = Dict[str, str]
"""
A single message with model-specific fields, such as `role` for OpenAI's models.
Must contain a `content` field.
"""

Conversation = List[Message]
"""A list of messages forming a conversation."""

AnnotatedConversation = List[AnnotatedMessage]
"""A list of messages with model-specific fields forming a conversation."""

Context = Union[
    Message,
    AnnotatedMessage,
    Conversation,
    AnnotatedConversation,
    List[Conversation],
    List[AnnotatedConversation],
    # Alternative representations
    npt.NDArray[
        np.str_
    ],  # can be equivalent to List[str] or List[List[str]], depending on the shape
    Dataset[
        str, str
    ],  # equivalent to List[List[str]], where the length of each inner list is 1
]
"""
The possible types of context input for the `generate` method and its derivatives.
Note that a list of messages (`List[Message]`) is equivalent to a single conversation (`Conversation`), and the same applies to annotated messages.
"""


class Model(ABC):
    """An abstract class for interacting with a model."""

    @dataclass(kw_only=True)
    class Config:
        """The configuration for the model."""

        name: str = MISSING
        """The name of the model."""

        default_max_tokens: int = 100
        """The default value for the maximum number of tokens to generate per context string."""

        train_batch_size: int = 64
        """The batch size for training purposes."""

        generate_batch_size: int = 64
        """The batch size for inference purposes."""

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        """The device which will be used to run the model."""

        use_progress_bar: bool = True
        """Whether to use a progress bar for long-running operations."""

        debug: bool = False
        """Whether to use debug-level logs."""

    def __init__(self, config: Config, logger: Optional[Logger] = None):
        """
        Initialize the model.

        ### Parameters
        ----------
        `config`: the configuration of the model.
        [optional] `logger`: the logger to be used.
        """

        self._config = config
        self._logger = logger

        if self._logger and config.debug:
            self._logger.debug({"[Model.config]": self._config.__dict__})

        if config.use_progress_bar:
            self._log_progress = log_progress
        else:
            self._log_progress = lambda x: x

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._config.name})"

    # TODO: wrap tokenizers in our own common interface class (belonging to this module)
    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """The tokenizer of the model."""
        pass

    def _parse_context(
        self,
        context: Context,
        unsafe: bool = False,
    ) -> List[List[Dict[str, str]]]:
        """
        Parses the context input.
        Check `generate`'s signature for allowed input types and their meanings.

        ### Parameters
        ----------
        `context`: the context to parse.
        `unsafe`: whether to bypass expensive input validations.

        ### Returns
        -------
        The parsed context, as a list of list of dictionaries.
        - If the length of the outer list is 1, then the context is a single message/conversation.

        ### Raises
        -------
        `ValueError`: if the input type is not supported.
        `ValueError`: if a message dictionary does not contain a `content` field.
        `AssertionError`: if a context list is provided and it is empty.
        """

        if isinstance(context, list):
            assert len(context) > 0, "the context list must not be empty."

        # Single message
        if isinstance(context, str):
            return [[{"content": context}]]
        elif isinstance(context, dict):  # with model-specific fields
            if "content" not in context:
                raise ValueError(
                    "The message dictionary must contain a `content` field."
                )
            return [[context]]

        # Single conversation
        elif (isinstance(context, np.ndarray) and context.ndim == 1) or (
            isinstance(context, list) and isinstance(context[0], str)
        ):
            return [[{"content": input} for input in context]]
        elif isinstance(context, list) and isinstance(
            context[0], dict
        ):  # with model-specific fields
            if not unsafe:
                for message in context:
                    if "content" not in message:
                        raise ValueError(
                            "All message dictionaries must contain a `content` field."
                        )
            return [context]

        # Multiple messages/conversations
        elif (isinstance(context, np.ndarray) and context.ndim == 2) or (
            isinstance(context, list)
            and isinstance(context[0], list)
            and isinstance(context[0][0], str)
        ):
            return [
                [{"content": message} for message in conversation]
                for conversation in context
            ]
        elif isinstance(context, Dataset):
            return [[{"content": input}] for input in context.test_set.inputs]
        elif (
            isinstance(context, list)
            and isinstance(context[0], list)
            and isinstance(context[0][0], dict)
        ):
            if not unsafe:
                for conversation in context:
                    for message in conversation:
                        if "content" not in message:
                            raise ValueError(
                                "All message dictionaries must contain a `content` field."
                            )
            return context

        raise ValueError(
            f"Invalid type for `context`: {type(context)}. Check the function's signature for allowed input types."
        )

    # TODO: return logprobs too
    def generate(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
        unsafe: bool = False,
    ) -> npt.NDArray[np.str_]:
        """
        Generates the next given number of tokens in the sequence.
        This method can be overriden by the child class to take advantage of GPU parallelization for multi-context inputs.

        ### Definitions
        ----------
        - A single independent message is the smallest unit of input.
            - Represented by a single string or dictionary.
            - Dictionaries allow to add model-specific fields, such as `role` for OpenAI's models.
            - Dictionaries can contain any number of fields, but the `content` field is required and contains the message's content.
        - A single conversation of dependent messages is a list of messages, from which only a single output is generated.
            - Represented by a list of strings/dictionaries.
        - Multiple messages/conversations yield multiple outputs.
            - Represented by a list of lists of strings/dictionaries.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `max_tokens`: the maximum number of tokens to generate per context string.
        `n_samples`: the number of samples to generate for each context string.
        `unsafe`: whether to bypass expensive input validations.

        ### Returns
        -------
        The generated tokens.
        - The return type is a `numpy.NDArray` of strings of shape (`len(context)`, `n_samples`).
        - If `context` is a single string/dictionary, then `len(context)` is 1.

        ### Raises
        -------
        `ValueError`: if the input type is not supported.
        `ValueError`: if a message dictionary does not contain a `content` field.
        `AssertionError`: if a context list is provided and it is empty.
        """

        context = self._parse_context(context, unsafe=unsafe)

        if self._logger:
            self._logger.info(
                f"[{self.__class__.__name__}] Generating {n_samples} samples for {len(context)} contexts"
            )

        outputs = np.array(
            [
                self._generate_impl(input, n_samples=n_samples, max_tokens=max_tokens)
                for input in context
            ]
        )

        return outputs

    def _call_impl(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _generate_impl(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
    ) -> npt.NDArray[np.str_]:
        """
        The model's internal implementation of `generate` acting on a single conversation (i.e., list of messages).

        ### Parameters
        ----------
        `context`: the context to generate from.
        - Each message (dictionary) must contain the `content` field.
        `n_samples`: the number of samples to generate for the context string.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If `None`, `config.default_max_tokens` will be used.

        ### Returns
        -------
        A `numpy.NDArray` with the generated tokens for each sample of shape (`n_samples`).
        """

        pass

    @abstractmethod
    def fine_tune(self, dataset: Union[Dataset, List[Tuple[str, str]]]):
        """
        Fine-tunes the model.

        ### Parameters
        ----------
        `dataset`: the dataset to fine-tune the model on.
        """

        pass
