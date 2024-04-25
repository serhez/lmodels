from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import torch

from lmodels.protocols import Dataset, Logger
from lmodels.utils import NullLogger

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


class Model(ABC):
    """An abstract class for interacting with a model."""

    @dataclass(kw_only=True)
    class Config:
        """The configuration for the model."""

        name: str
        """The name of the model."""

        default_max_tokens: int = 100
        """The default value for the maximum number of tokens to generate per context string."""

        train_batch_size: int = 64
        """The batch size for training purposes."""

        generate_batch_size: int = 64
        """The batch size for inference purposes."""

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        """The device which will be used to run the model."""

        calls_threshold: int | None = 100
        """
        The maximum number of calls to the model's forward pass.
        This safeguard can be used to prevent infinite loops and other unexpected behavior.
        It can be set to `None` to disable the safeguard.
        """

    @property
    @abstractmethod
    def config_cls(self) -> type[Config]:
        """The configuration class of the model."""

        ...

    def __init__(self, config: Config, logger: Logger | None = None):
        """
        Initialize the model.

        ### Parameters
        ----------
        `config`: the configuration of the model.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        self._config = config
        if logger is None:
            self._logger = NullLogger()
        else:
            self._logger = logger
        self._stats = {
            "n_tokens_context": 0,
            "n_tokens_output": 0,
            "n_calls": 0,
        }

        self._logger.debug(
            {f"[{self.__class__.__name__}.config]": asdict(self._config)}
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._config.name})"

    # TODO: wrap tokenizers in our own common interface class (belonging to this module)
    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """The tokenizer of the model."""

        ...

    @property
    def usage(self) -> dict[str, Any]:
        """
        The usage statistics of the model, containing:
        - `n_tokens_context`: the sum of the number of tokens which each sample's context has.
        - `n_tokens_output`: the sum of the number of tokens which each sample has generated.
        - `n_calls`: the number of calls to the model's forward pass.
        """

        return self._stats

    def _record_model_usage(self, stats: dict[str, Any]):
        if "n_tokens_context" in stats:
            self._stats["n_tokens_context"] += stats["n_tokens_context"]
        if "n_tokens_output" in stats:
            self._stats["n_tokens_output"] += stats["n_tokens_output"]
        if "n_calls" in stats:
            self._stats["n_calls"] += stats["n_calls"]

    # TODO: return logprobs too
    def generate(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: int | None = None,
        unsafe: bool = False,
    ) -> tuple[npt.NDArray[np.str_], dict[str, Any]]:
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
        A tuple containing:
        - A `numpy.NDArray` of strings of shape (`len(context)`, `n_samples`).
            - If `context` is a single string/dictionary, then `len(context)` is 1.
        - A dictionary with usage statistics including (but not limited to, depending on the model):
            - `n_tokens_context`: the sum of the number of tokens which each sample's context has.
            - `n_tokens_output`: the sum of the number of tokens which each sample has generated.
            - `n_calls`: the number of calls to the model's forward pass.

        ### Raises
        -------
        `AssertionError`: if the number of calls to the model's forward pass is expected to exceed the configured threshold.
        `ValueError`: if the input type is not supported.
        `ValueError`: if a message dictionary does not contain a `content` field.
        `AssertionError`: if a context list is provided and it is empty.
        """

        if isinstance(context, np.ndarray):
            expected_n_calls = len(context)
        elif isinstance(context, list):
            expected_n_calls = len(context)
        elif isinstance(context, Dataset):
            expected_n_calls = len(context.test_set.inputs)
        else:
            expected_n_calls = 1

        expected_n_calls *= n_samples

        if self._config.calls_threshold:
            assert (
                self.usage["n_calls"] + expected_n_calls <= self._config.calls_threshold
            ), f"Number of calls to the model's forward pass are expected to exceed the configured threshold of `Config.calls_threshold={self._config.calls_threshold}`."

        return self._generate_batch(
            context, n_samples=n_samples, max_tokens=max_tokens, unsafe=unsafe
        )

    def _parse_context(
        self,
        context: Context,
        unsafe: bool = False,
    ) -> list[list[dict[str, str]]]:
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

    def _generate_batch(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: int | None = None,
        unsafe: bool = False,
    ) -> tuple[npt.NDArray[np.str_], dict[str, Any]]:
        """
        Internal method for generating samples in batches.
        This method can be overriden by the child class to take advantage of GPU parallelization for multi-context inputs.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `max_tokens`: the maximum number of tokens to generate per context string.
        `n_samples`: the number of samples to generate for each context string.
        `unsafe`: whether to bypass expensive input validations.

        ### Returns
        -------
        A tuple containing:
        - A `numpy.NDArray` of strings of shape (`len(context)`, `n_samples`).
            - If `context` is a single string/dictionary, then `len(context)` is 1.
        - A dictionary with usage statistics including (but not limited to, depending on the model):
            - `n_tokens_context`: the sum of the number of tokens which each sample's context has.
            - `n_tokens_output`: the sum of the number of tokens which each sample has generated.
            - `n_calls`: the number of calls to the model's forward pass.

        ### Raises
        -------
        `ValueError`: if the input type is not supported.
        `ValueError`: if a message dictionary does not contain a `content` field.
        `AssertionError`: if a context list is provided and it is empty.
        """

        context = self._parse_context(context, unsafe=unsafe)

        self._logger.info(
            f"[{self.__class__.__name__}] Generating {n_samples} samples for {len(context)} contexts"
        )

        outputs, ind_stats = zip(
            *[
                self._generate_single(input, n_samples=n_samples, max_tokens=max_tokens)
                for input in context
            ]
        )
        outputs = np.array(list(outputs))
        ind_stats = list(ind_stats)

        # Common usage statistics
        agg_stats: dict[str, Any] = {
            "n_tokens_context": sum([stats["n_tokens_context"] for stats in ind_stats]),
            "n_tokens_output": sum([stats["n_tokens_output"] for stats in ind_stats]),
            "n_calls": sum([stats["n_calls"] for stats in ind_stats]),
        }

        # Model-specific usage statistics
        for key in ind_stats[0]:
            if key not in agg_stats:
                agg_stats[key] = [stats[key] for stats in ind_stats]

        self._logger.debug(
            {
                f"[{self.__class__.__name__}.generate]": None,
                "Context": context,
                "Outputs": outputs,
                "N. samples": n_samples,
                "Max. tokens": max_tokens,
                "Usage stats.": agg_stats,
            }
        )

        return outputs, agg_stats

    def _call_impl(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
    ) -> tuple[npt.NDArray[np.str_], dict[str, Any]]:
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
        A tuple containing:
        - A `numpy.NDArray` with the generated tokens for each sample of shape (`n_samples`).
        - A dictionary with usage statistics including (but not limited to, depending on the model):
            - `n_tokens_context`: the sum of the number of tokens which each sample's context has.
            - `n_tokens_output`: the sum of the number of tokens which each sample has generated.
            - `n_calls`: the number of calls to the model's forward pass.
        """

        ...

    @abstractmethod
    def fine_tune(self, dataset: Dataset | list[tuple[str, str]]):
        """
        Fine-tunes the model.

        ### Parameters
        ----------
        `dataset`: the dataset to fine-tune the model on.
        """

        ...
