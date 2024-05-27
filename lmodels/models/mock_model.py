import os
import re
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import requests
import transformers
from lcommon.protocols import Logger
from lcommon.types import AnnotatedConversation
from lcommon.utils import Usage, classproperty

from lmodels import Model


@dataclass(kw_only=True)
class MockResponse:
    context: str
    """A regex matching the input context string."""

    outputs: list[tuple[str, float]]
    """
    A list of possible tuples (output, probability).
    The probabilities must add up to 1.0.
    """

    def __post_init__(self):
        """Checks if the probabilities add up to 1.0."""

        sum = np.sum([prob for _, prob in self.outputs])
        if not np.isclose(sum, 1.0):
            raise ValueError(
                f"The probabilities must add up to 1.0, but they add up to {sum}"
            )


@dataclass(kw_only=True)
class MockResponseCollection:
    matching: list[MockResponse] = field(default_factory=list)

    """A list of possible responses to use matching a given context."""

    default: list[tuple[str, float]] | None = None
    """
    The default responses to use if no regex matches.
    Each response is a tuple (output, prob); the probs must add up to 1.0.
    If not provided, a random response will be generated.
    """

    def __post_init__(self):
        """Checks if the probabilities add up to 1.0."""

        if self.default:
            sum = np.sum([prob for _, prob in self.default])
            if not np.isclose(sum, 1.0):
                raise ValueError(
                    f"The probabilities for the default responses must add up to 1.0, but they add up to {sum}"
                )


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

        delete_word_cache: bool = False
        """Whether to delete the file containing the word cache upon the object's destruction."""

        responses: MockResponseCollection = field(
            default_factory=MockResponseCollection
        )
        """The possible responses for the model."""

        def __post_init__(self):
            if type(self.responses) is not MockResponseCollection:
                try:
                    self.responses = MockResponseCollection(
                        matching=[
                            MockResponse(
                                context=response.context,
                                outputs=[tuple(o) for o in response.outputs],  # type: ignore[reportArgumentType]
                            )
                            for response in self.responses.matching
                        ],
                        default=None
                        if self.responses.default is None
                        else [tuple(o) for o in self.responses.default],
                    )
                except Exception as e:
                    raise ValueError(
                        "The responses must be a MockResponseCollection or a dictionary with the keys 'matching' and 'default'."
                    ) from e

    @classproperty
    def config_cls(cls) -> type[Config]:
        return cls.Config

    @classproperty
    def generation_info_cls(cls) -> type[Model.GenerationInfo]:
        return cls.GenerationInfo

    _WORDS_CACHE_PATH = "mock_model_cached_words.txt"
    _WORDS_URL = "https://www.mit.edu/~ecprice/wordlist.10000"

    def __init__(
        self,
        config: Config,
        logger: Logger | None = None,
    ):
        """
        Initializes the mock model with the given configuration.

        ### Parameters
        ----------
        `config`: the configuration for the mock model.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.

        ### Raises
        ------
        `ValueError`: the `probs` do not add up to 1 or do not have the same length as the `outputs`.
        """

        super().__init__(config, logger)
        self._config: MockModel.Config  # pyright is too dumb to understand this type

        if os.path.isfile(self._WORDS_CACHE_PATH):
            return

        # Create a list of words
        response = requests.get(self._WORDS_URL)
        all_words = response.content.splitlines()

        # Remove spaces from the words
        all_words = [word.decode("utf-8") for word in all_words]
        all_words = [word for word in all_words if all(char.isalpha() for char in word)]

        # Cache the words into a file
        with open(self._WORDS_CACHE_PATH, "w") as file:
            file.write("\n".join(all_words))

    def __del__(self):
        """Deletes the cached words."""

        if self._config.delete_word_cache:
            try:
                os.remove(self._WORDS_CACHE_PATH)
            except FileNotFoundError:
                pass

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        raise NotImplementedError("The mock model does not have a tokenizer.")

    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
    ) -> tuple[npt.NDArray[np.str_], Model.GenerationInfo]:
        if max_tokens is None:
            max_tokens = self._config.max_tokens

        input = "\n".join([message["content"] for message in context])
        output = None

        # A matching response
        for response in self._config.responses.matching:
            if re.match(response.context, input):
                output = np.array(
                    [
                        np.random.choice(
                            [output for output, _ in response.outputs],
                            p=[prob for _, prob in response.outputs],
                        )[:max_tokens]
                        for _ in range(n_samples)
                    ]
                )

        # A default response
        if output is None and self._config.responses.default:
            output = np.array(
                [
                    np.random.choice(
                        [output for output, _ in self._config.responses.default],
                        p=[prob for _, prob in self._config.responses.default],
                    )[:max_tokens]
                    for _ in range(n_samples)
                ]
            )

        # A random response
        if output is None:
            with open(self._WORDS_CACHE_PATH, "r") as file:
                words = [
                    line[:-1] for line in file.readlines()
                ]  # Remove the newline character
            output = np.array(
                [
                    " ".join(np.random.choice(words, size=max_tokens))
                    for _ in range(n_samples)
                ]
            )

        info = Model.GenerationInfo(
            usage=Usage(
                n_calls=n_samples,
                n_tokens_context=sum([len(m["content"].split()) for m in context]),
                n_tokens_output=sum([len(o.split()) for o in output]),
            )
        )
        self.usage += info.usage

        return output, info

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the mock model.")
