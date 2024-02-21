from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from ldata import Dataset  # TODO: Drop this dependency with own Dataset interface
from mloggers import Logger


class Model(ABC):
    """An abstract class for interacting with a model."""

    @dataclass(kw_only=True)
    class Config:
        """The configuration for the model."""

        name: str = MISSING
        """The name of the model."""

        train_batch_size: int = 64
        """The batch size for training purposes."""

        generate_batch_size: int = 64
        """The batch size for inference purposes."""

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        """The device which will be used to run the model."""

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

        if self._logger and self._config.debug:
            self._logger.debug({"Model config": config})

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._config.name})"

    # TODO: wrap tokenizers in our own common interface class (belonging to this module)
    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """The tokenizer of the model."""
        pass

    # TODO: progress bar for long generations
    def generate(
        self,
        context: Union[
            str,
            List[str],
            Iterator[str],
            Dataset[str, str],
        ],
        max_tokens: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """
        Generates the next given number of tokens in the sequence.
        It has similar functionality to HuggingFace's `pipeline` method.
        This method can be overriden by the child class to take advantage of GPU parallelization for multi-context inputs.

        ### Parameters
        ----------
        `context`: the context/s to generate from.
        - If it is a `Dataset`, the model will generate from all samples in the test set.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If None, the model will generate tokens until the EOS token is produced.

        ### Returns
        -------
        The generated tokens.
        - If `context` is a string, the return value is a string.
        - If `context` is a list, iterator of strings or `Dataset`, the return value is a list of strings.
        """

        if isinstance(context, str):
            return self._generate_impl(context, max_tokens)
        elif isinstance(context, Dataset):
            context = list(context.test_set.inputs)
        elif not isinstance(context, list) and not isinstance(context, Iterator):
            raise ValueError(
                f"Invalid type for `context`: {type(context)}. Must be a string, list of strings, iterator returning strings or `Dataset`."
            )

        outputs = []
        for i, c in enumerate(context):
            if self._logger:
                self._logger.info(
                    f"[{self.__class__.__name__}] Generating {i}/{len(context)}"
                )
            outputs.append(self._generate_impl(c, max_tokens))

        return outputs

    def _call_impl(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _generate_impl(self, context: str, max_tokens: Optional[int] = None) -> str:
        """
        The model's internal implementation of `generate` acting on a single context string.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If None, the model will generate tokens until the EOS token is produced.

        ### Returns
        -------
        The generated tokens.
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
