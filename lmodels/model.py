from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from ldata import Dataset  # TODO: Drop this dependency with own Dataset interface
from mloggers import Logger
from mloggers.progress import log_progress


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

    def generate(
        self,
        context: Union[
            str,
            List[str],
            npt.NDArray[np.str_],
            Iterator[str],
            Dataset[str, str],
        ],
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
    ) -> npt.NDArray[np.str_]:
        """
        Generates the next given number of tokens in the sequence.
        It has similar functionality to HuggingFace's `pipeline` method.
        This method can be overriden by the child class to take advantage of GPU parallelization for multi-context inputs.

        ### Parameters
        ----------
        `context`: the context/s to generate from.
        - If it is a `Dataset`, the model will generate from all samples in the test set.
        `max_tokens`: the maximum number of tokens to generate per context string.
        `n_samples`: the number of samples to generate for each context string.

        ### Returns
        -------
        The generated tokens.
        - The return type is a `numpy.NDArray` of strings of size [`len(context)`, `n_samples`]; if `context` is a single string, then `len(context)` is 1.
        """

        if isinstance(context, str):
            return np.array([self._generate_impl(context, max_tokens, n_samples)])
        elif isinstance(context, Dataset):
            context = list(context.test_set.inputs)
        elif isinstance(context, Iterator) or isinstance(context, np.ndarray):
            context = list(context)
        elif not isinstance(context, list):
            raise ValueError(
                f"Invalid type for `context`: {type(context)}. Must be a string, list of strings, iterator returning strings or `Dataset`."
            )

        outputs = np.zeros((len(context), n_samples), dtype=np.str_)
        if self._logger:
            self._logger.info(
                f"[{self.__class__.__name__}] Generating {n_samples} samples for {len(context)} contexts"
            )
        for i in log_progress(range(len(context))):
            c_outputs = []
            for _ in range(n_samples):
                c_outputs.append(self._generate_impl(context[i], max_tokens))
            outputs.append(c_outputs)

        return outputs

    def _call_impl(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    @abstractmethod
    def _generate_impl(
        self, context: str, n_samples: int = 1, max_tokens: Optional[int] = None
    ) -> npt.NDArray[np.str_]:
        """
        The model's internal implementation of `generate` acting on a single context string.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `n_samples`: the number of samples to generate for the context string.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If `None`, `config.default_max_tokens` will be used.

        ### Returns
        -------
        A `numpy.NDArray` with the generated tokens for each sample.
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
