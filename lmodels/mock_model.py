import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import requests
import transformers
from mloggers import Logger

from lmodels.model import Model


class MockModel(Model):
    """
    A model that generates random strings for testing purposes.
    It is also possible to provide a list of pre-determined output sequences with an associated probability distribution.
    """

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the mock model."""

        name: str = "MockModel"
        """The name of the model."""

        outputs: Optional[List[str]] = None
        """
        The list of possible outputs to use.
        - If not provided, sequences of random tokens of at most `max_tokens` length will be output.
        """

        probs: Optional[List[float]] = None
        """
        The probabilities for each of the `outputs`."
        - If provided, they must add up to 1 and have the same length as the `outputs`.
        - If `outputs` are not provided, this parameter will be ignored.
        - If not provided, a uniform distribution will be used to sample the `outputs` (if they are given).
        """

    def __init__(
        self,
        config: Config,
        logger: Optional[Logger] = None,
    ):
        """
        Initializes the mock model with the given configuration.

        ### Parameters
        ----------
        `config`: the configuration for the mock model.
        [optional] `logger`: the logger to be used.

        ### Raises
        ------
        `ValueError`: the `probs` do not add up to 1 or do not have the same length as the `outputs`.
        """

        super().__init__(config, logger)

        if config.outputs is not None and config.probs is not None:
            if len(config.outputs) != len(config.probs):
                raise ValueError(
                    f"The lengths of `outputs` ({len(config.outputs)} and `probs` ({len(config.probs)}) must be the same"
                )
            elif np.sum(config.probs) != 1.0:
                raise ValueError(
                    f"`probs` add up to {np.sum(config.probs)}; they must add up to 1.0"
                )

        self._outputs = config.outputs
        self._probs = config.probs

        if not self._outputs:
            if self._logger:
                self._logger.warning(
                    "No outputs were provided. The model will generate random sequences."
                )

            # Create a list of words
            response = requests.get("https://www.mit.edu/~ecprice/wordlist.10000")
            all_words = response.content.splitlines()

            # Remove spaces from the words
            all_words = [word.decode("utf-8") for word in all_words]
            all_words = [
                word for word in all_words if all(char.isalpha() for char in word)
            ]

            # Cache the words into a file
            with open("mock_model_cached_words.txt", "w") as file:
                file.write("\n".join(all_words))

    def __del__(self):
        """Deletes the cached words."""

        try:
            os.remove("mock_model_cached_words.txt")
        except FileNotFoundError:
            pass

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        raise NotImplementedError("The mock model does not have a tokenizer.")

    def _generate_impl(
        self, _, n_samples: int = 1, max_tokens: Optional[int] = None
    ) -> npt.NDArray[np.str_]:
        if max_tokens is None:
            max_tokens = self._config.default_max_tokens

        if self._outputs:
            output = np.random.choice(self._outputs, size=(n_samples,), p=self._probs)
        else:
            with open("mock_model_cached_words.txt", "r") as file:
                words = file.readlines()
            output = np.array(
                [
                    " ".join(np.random.choice(words, size=max_tokens))
                    for _ in range(n_samples)
                ]
            )

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[MockModel._generate_impl]": None,
                    "Output": output,
                    "n_samples": n_samples,
                    "max_tokens": max_tokens,
                }
            )

        return output

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the mock model.")
