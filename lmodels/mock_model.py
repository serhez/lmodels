import random
from dataclasses import dataclass

from lmodels.model import Model


class MockModel(Model):
    """
    A model that generates random strings for testing purposes.
    """

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the mock model."""

        name = "MockModel"
        """The name of the model."""

        max_tokens: int = 100
        """The default value for the maximum number of tokens to generate per context string."""

    def __init__(self, config: Config):
        """
        Initializes the mock model with the given configuration.

        Parameters
        ----------
        `config`: the configuration for the mock model.
        """

        super().__init__(config)

    @property
    def tokenizer(self) -> None:
        return None

    def _generate_impl(self, _, max_tokens: int) -> str:
        return "".join(
            random.choices(
                "abcdefghijklmnopqrstuvwxyz ", k=max_tokens or self._config.max_tokens
            )
        )

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the mock model.")
