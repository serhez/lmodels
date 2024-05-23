from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from lmodels.protocols import Dataset, Logger
from lmodels.types import AnnotatedConversation, Context
from lmodels.utils import NullLogger, Usage, classproperty


class Model(ABC):
    """An abstract class for interacting with a model."""

    _DEFAULT_ROLE = "user"

    @dataclass(kw_only=True)
    class Config:
        """The configuration for the model."""

        name: str
        """The name of the model."""

        max_tokens: int = 100
        """The default value for the maximum number of tokens to generate per context string."""

        train_batch_size: int = 64
        """The batch size for training purposes."""

        generate_batch_size: int = 64
        """The batch size for inference purposes."""

        calls_threshold: int | None = 100
        """
        The maximum number of calls to the model's forward pass.
        This safeguard can be used to prevent infinite loops and other unexpected behavior.
        It can be set to `None` to disable the safeguard.
        """

    @dataclass(kw_only=True)
    class GenerationInfo:
        """Extra information about the generation process of the model."""

        usage: Usage = field(default_factory=Usage)
        """The usage statistics of the model."""

        def __add__(self, other: Model.GenerationInfo | None) -> Model.GenerationInfo:
            """
            Combines the generation information of two models.

            ### Parameters
            ----------
            `other`: the other generation information to combine with.

            ### Returns
            ----------
            The combined generation information.
            """

            return Model.GenerationInfo(
                usage=self.usage + other.usage if other is not None else self.usage
            )

        def __radd__(self, other: Model.GenerationInfo | None) -> Model.GenerationInfo:
            """
            Combines the generation information of two models.

            ### Parameters
            ----------
            `other`: the other generation information to combine with.

            ### Returns
            ----------
            The combined generation information.
            """

            return self + other

        def __iadd__(self, other: Model.GenerationInfo | None) -> Model.GenerationInfo:
            """
            Combines the generation information of two models in-place.

            ### Parameters
            ----------
            `other`: the other generation information to combine with.

            ### Returns
            ----------
            The combined generation information.
            """

            if other is not None:
                self.usage += other.usage

            return self

        def to_json(self) -> dict[str, Any]:
            """
            Converts the generation information to a JSON-serializable dictionary.

            ### Returns
            ----------
            The JSON-serializable dictionary.
            """

            return {
                "usage": self.usage.to_json(),
            }

    @classproperty
    @abstractmethod
    def config_cls(cls) -> type[Config]:
        """The configuration class of the model."""

        ...

    @classproperty
    @abstractmethod
    def generation_info_cls(cls) -> type[GenerationInfo]:
        """The generation information class of the model."""

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
        self._usage = Usage()

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
    def usage(self) -> Usage:
        """The aggregated usage statistics of the model instance, accounting for all generations."""

        return self._usage

    @usage.setter
    def usage(self, value: Usage):
        """Sets the usage statistics of the model instance."""

        self._usage = value

    def _parse_context(
        self,
        context: Context,
    ) -> list[AnnotatedConversation]:
        """
        Parses the context input and returns it as a list of `AnnotatedConversation` objects.

        ### Parameters
        ----------
        `context`: the context to parse.

        ### Returns
        -------
        The parsed context as a list of `AnnotatedConversation` objects.
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
            return [[{"content": context, "role": self._DEFAULT_ROLE}]]
        elif isinstance(context, dict):  # with model-specific fields
            if "content" not in context:
                raise ValueError(
                    "The message dictionary must contain a `content` field."
                )
            if "role" not in context:
                context["role"] = self._DEFAULT_ROLE
            return [[context]]

        # Single conversation
        elif (
            (isinstance(context, np.ndarray) and context.ndim == 1)
            or isinstance(context, list)
        ) and all(isinstance(context[i], str) for i in range(len(context))):
            return [
                [{"content": input, "role": self._DEFAULT_ROLE} for input in context]
            ]  # type: ignore[reportReturnType]
        elif (
            (isinstance(context, np.ndarray) and context.ndim == 1)
            or isinstance(context, list)
        ) and all(
            isinstance(context[i], dict) for i in range(len(context))
        ):  # with model-specific fields
            for message in context:
                if "content" not in message:
                    raise ValueError(
                        "All message dictionaries must contain a `content` field."
                    )
                if "role" not in message:
                    message["role"] = self._DEFAULT_ROLE  # type: ignore
            return [context]  # type: ignore[reportReturnType]

        # Multiple messages/conversations
        elif (
            (
                (isinstance(context, np.ndarray) and context.ndim == 2)
                or isinstance(context, list)
            )
            and all(
                (
                    (isinstance(context[i], np.ndarray) and context[i].ndim == 1)  # type: ignore[reportAttributeAccessIssue]
                    or isinstance(context[i], list)
                )
                for i in range(len(context))
            )
            and all(
                isinstance(context[i][j], str)  # type: ignore[reportArgumentType]
                for i, j in np.ndindex((len(context), len(context[0])))
            )
        ):
            return [  # type: ignore[reportReturnType]
                [
                    {"content": message, "role": self._DEFAULT_ROLE}
                    for message in conversation
                ]
                for conversation in context
            ]
        elif (
            (
                (isinstance(context, np.ndarray) and context.ndim == 2)
                or isinstance(context, list)
            )
            and all(
                (
                    (isinstance(context[i], np.ndarray) and context[i].ndim == 1)  # type: ignore[reportAttributeAccessIssue]
                    or isinstance(context[i], list)
                )
                for i in range(len(context))
            )
            and all(
                isinstance(context[i][j], dict)  # type: ignore[reportArgumentType]
                for i, j in np.ndindex((len(context), len(context[0])))
            )
        ):
            for conversation in context:
                for message in conversation:
                    if "content" not in message:
                        raise ValueError(
                            "All message dictionaries must contain a `content` field."
                        )
                    if "role" not in message:
                        message["role"] = self._DEFAULT_ROLE  # type: ignore[reportIndexIssue]
            return context  # type: ignore[reportReturnType]
        elif isinstance(context, Dataset):
            return [
                [{"content": input, "role": self._DEFAULT_ROLE}]
                for input in context.test_set.inputs
            ]

        raise ValueError(
            f"Invalid type for `context`: {type(context)}. Check the function's signature for allowed input types."
        )

    # TODO: return logprobs too
    def generate(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.str_], GenerationInfo]:
        """
        Generates the next tokens in the sequence given the context.

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

        ### Returns
        -------
        A tuple containing:
        - A `numpy.NDArray` of strings of shape (`len(context)`, `n_samples`).
            - If `context` is a single string/dictionary, then `len(context)` is 1.
        - Extra information about the generation process of the model.

        ### Raises
        -------
        `Exception`: any exception raised by the model's internal implementation; consult the wrapped model's documentation for more information.
        - This includes, e.g., errors for surpassing the context size, exceeding the credits available in your account for paid-for services, server errors, etc.
        - You should handle these exceptions in your application.
        `AssertionError`: if the number of calls to the model's forward pass is expected to exceed the configured threshold.
        `ValueError`: if the input type is not supported.
        `ValueError`: if a message dictionary does not contain a `content` field.
        `AssertionError`: if a context list is provided and it is empty.
        """

        if isinstance(context, np.ndarray):
            expected_n_calls = len(context) * n_samples
        elif isinstance(context, list):
            expected_n_calls = len(context) * n_samples
        elif isinstance(context, Dataset):
            expected_n_calls = len(context.test_set.inputs) * n_samples
        else:
            expected_n_calls = n_samples

        if self._config.calls_threshold:
            assert (
                self.usage.n_calls + expected_n_calls <= self._config.calls_threshold
            ), f"Number of calls to the model's forward pass are expected to exceed the configured threshold of `Config.calls_threshold={self._config.calls_threshold}`."

        return self._generate_batch(
            self._parse_context(context),
            n_samples=n_samples,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _call_impl(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def _generate_batch(
        self,
        context: list[AnnotatedConversation],
        n_samples: int = 1,
        max_tokens: int | None = None,
        **kwargs,
    ) -> tuple[npt.NDArray[np.str_], GenerationInfo]:
        """
        Internal method for generating samples in batches.
        This method can be overriden by the child class to take advantage of GPU parallelization for multi-context inputs.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `max_tokens`: the maximum number of tokens to generate per context string.
        `n_samples`: the number of samples to generate for each context string.
        Other keyword arguments can be passed to the model's generation method, given the specific model.

        ### Returns
        -------
        A tuple containing:
        - A `numpy.NDArray` of strings of shape (`len(context)`, `n_samples`).
            - If `context` is a single string/dictionary, then `len(context)` is 1.
        - Extra information about the generation process of the model.

        ### Raises
        -------
        `ValueError`: if the input type is not supported.
        `ValueError`: if a message dictionary does not contain a `content` field.
        `AssertionError`: if a context list is provided and it is empty.
        """

        self._logger.info(
            f"[{self.__class__.__name__}] Generating {n_samples} samples for {len(context)} contexts"
        )

        outputs, ind_info = zip(
            *[
                self._generate_single(
                    input, n_samples=n_samples, max_tokens=max_tokens, **kwargs
                )
                for input in context
            ]
        )
        outputs = np.array(list(outputs))
        ind_info = list(ind_info)

        # Common usage statistics
        agg_info = sum(ind_info, start=self.generation_info_cls())

        self._logger.debug(
            {
                f"[{self.__class__.__name__}.generate]": None,
                "Context": context,
                "Outputs": outputs,
                "N. samples": n_samples,
                "Max. tokens": max_tokens,
                "Info": agg_info,
            }
        )

        return outputs, agg_info

    @abstractmethod
    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
        **kwargs,
    ) -> tuple[npt.NDArray[np.str_], GenerationInfo]:
        """
        The model's internal implementation of `generate` acting on a single conversation (i.e., list of messages).

        ### Parameters
        ----------
        `context`: the context to generate from.
        - Each message (dictionary) must contain the `content` field.
        `n_samples`: the number of samples to generate for the context string.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If `None`, `config.max_tokens` will be used.
        Other keyword arguments can be passed to the model's generation method, given the specific model.

        ### Returns
        -------
        A tuple containing:
        - A `numpy.NDArray` with the generated tokens for each sample of shape (`n_samples`).
        - Extra information about the generation process of the model.
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
