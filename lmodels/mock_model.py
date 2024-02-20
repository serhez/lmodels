import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import requests
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

        max_tokens: int = 100
        """The default value for the maximum number of tokens to generate per context string."""

    def __init__(
        self,
        config: Config,
        logger: Optional[Logger] = None,
        outputs: Optional[List[str]] = None,
        probs: Optional[List[float]] = None,
    ):
        """
        Initializes the mock model with the given configuration.

        ### Parameters
        ----------
        `config`: the configuration for the mock model.
        [optional] `logger`: the logger to be used.
        [optional] `outputs`: the list of possible outputs to use.
        - If not provided, sequences of random tokens of at most `config.max_tokens` length will be output.
        [optional] `probs`: the probabilities for each of the `outputs`.
        - If provided, they must add up to 1 and have the same length as the `outputs`.
        - If `outputs` are not provided, this parameter will be ignored.
        - If not provided, a uniform distribution will be used to sample the `outputs` (if they are given).

        ### Raises
        ------
        `ValueError`: the `probs` do not add up to 1 or do not have the same length as the `outputs`.
        """

        super().__init__(config, logger)

        if outputs is not None and probs is not None:
            if len(outputs) != len(probs):
                raise ValueError(
                    f"The lengths of `outputs` ({len(outputs)} and `probs` ({len(probs)}) must be the same"
                )
            elif np.sum(probs) != 1.0:
                raise ValueError(
                    f"`probs` add up to {np.sum(probs)}; they must add up to 1.0"
                )

        self._outputs = outputs
        self._probs = probs

        if not outputs:
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
    def tokenizer(self) -> None:
        return None

    def _generate_impl(self, _, max_tokens: int) -> str:
        if self._outputs:
            output = np.random.choice(self._outputs, p=self._probs)
        else:
            with open("mock_model_cached_words.txt", "r") as file:
                words = file.readlines()
            output = " ".join(np.random.choice(words, size=max_tokens))

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[MockModel.generate]": None,
                    "Output": output,
                }
            )

        return output

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the mock model.")


try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_mock_model", node=MockModel.Config)
except ModuleNotFoundError:
    pass
