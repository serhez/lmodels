from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx
import numpy as np
import numpy.typing as npt
import transformers

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    raise ImportError("You must install the `openai` package to use the OpenAI models.")

from lmodels.model import AnnotatedConversation, Model
from lmodels.protocols import Logger
from lmodels.utils import Usage, classproperty


class OpenAIModel(Model):
    """
    An API wrapper for interacting with OpenAI models.

    Your API key must be stored in the environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` (for Azure API).
    If you are using the Azure API, you must also set the `AZURE_OPENAI_ENDPOINT` and `OPENAI_API_VERSION` environment variables.
    Other environment variables that can optionally be set are:
    - `OPENAI_ORG_ID`
    - `AZURE_OPENAI_AD_TOKEN`

    The default role for messages is "user" if none is provided when using `generate` via annotated messages (i.e., dictionaries).
    """

    _DEFAULT_ROLE = "user"  # if modified, reflect on the docstring above and any other relevant documentation

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the OpenAI model."""

        name: str = "OpenAIModel"
        """The name of the model."""

        use_azure: bool = False
        """Whether to use the Azure OpenAI API instead of the regular one."""

        architecture: str
        """
        The name of the model architecture to use.
        Must be listed as an OpenAI model architecture.
        """

        temperature: float = 1.0
        """The temperature for sampling from the model."""

        top_p: float = 1.0
        """The cumulative probability for nucleus sampling."""

        url_replacements: dict[str, str] = field(default_factory=lambda: dict())
        """
        A dictionary of URL replacements to be made for the Azure API url.
        The keys are the patterns to be identified in the `URL.path`, and the values are the whole new `URL.path` to be used instead.
        """

    @dataclass(kw_only=True)
    class GenerationInfo(Model.GenerationInfo):
        """The generation information for the OpenAI model."""

        finish_reasons: list[list[str]] = field(default_factory=lambda: list(list()))
        """The reasons for finishing the generation, for each output and sample."""

        def __add__(
            self, other: OpenAIModel.GenerationInfo
        ) -> OpenAIModel.GenerationInfo:
            return OpenAIModel.GenerationInfo(
                usage=self.usage + other.usage,
                finish_reasons=self.finish_reasons + other.finish_reasons,
            )

        def __iadd__(
            self, other: OpenAIModel.GenerationInfo
        ) -> OpenAIModel.GenerationInfo:
            self.usage += other.usage
            self.finish_reasons += other.finish_reasons
            return self

    @classproperty
    def config_cls(cls) -> type[Config]:
        return cls.Config

    @classproperty
    def generation_info_cls(cls) -> type[OpenAIModel.GenerationInfo]:
        return cls.GenerationInfo

    def __init__(self, config: Config, logger: Logger | None = None):
        """
        Initializes the OpenAI model.
        Your API key should be stored in the environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` (for Azure API).
        If you are using the Azure API, you must also set the `AZURE_OPENAI_ENDPOINT` and `OPENAI_API_VERSION` environment variables.
        Other environment variables that can be set are:
        - `OPENAI_ORG_ID`.
        - `AZURE_OPENAI_AD_TOKEN`.

        ### Parameters
        ----------
        `config`: the configuration for the OpenAI model.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        super().__init__(config, logger)

        self._config = config

        def _update_base_url(request: httpx.Request) -> None:
            for key, value in config.url_replacements.items():
                if key in request.url.path:
                    request.url = request.url.copy_with(path=value)

        if config.use_azure:
            assert (
                "AZURE_OPENAI_API_KEY" in os.environ
            ), "you must set the `AZURE_OPENAI_API_KEY` environment variable for `config.use_azure = True`"
            assert (
                "AZURE_OPENAI_ENDPOINT" in os.environ
            ), "you must set the `AZURE_OPENAI_ENDPOINT` environment variable for `config.use_azure = True`"

            self._client = AzureOpenAI(
                default_headers={
                    "Ocp-Apim-Subscription-Key": os.environ["AZURE_OPENAI_API_KEY"],
                },
                http_client=httpx.Client(
                    event_hooks={
                        "request": [_update_base_url],
                    }
                ),
            )
        else:
            assert (
                "OPENAI_API_KEY" in os.environ
            ), "you must set the `OPENAI_API_KEY` environment variable for `config.use_azure = False`"

            self._client = OpenAI()

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        raise NotImplementedError(
            "The OpenAI model does not currently provide a tokenizer."
        )

    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
    ) -> tuple[npt.NDArray[np.str_], GenerationInfo]:
        if max_tokens is None:
            max_tokens = self._config.default_max_tokens

        for message in context:
            if "role" not in message:
                message["role"] = self._DEFAULT_ROLE

        output = self._client.chat.completions.create(
            messages=context,
            model=self._config.architecture,
            max_tokens=max_tokens,
            n=n_samples,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
        )

        info = OpenAIModel.GenerationInfo(
            usage=Usage(
                n_calls=1,
                n_tokens_context=output.usage.prompt_tokens
                if output.usage is not None
                else 0,
                n_tokens_output=output.usage.completion_tokens
                if output.usage is not None
                else 0,
            ),
            finish_reasons=[[c.finish_reason for c in output.choices]]
            if output.choices is not None
            else [[]],
        )
        self.usage += info.usage

        for c in output.choices:
            if c.message.content is None:
                self._logger.warn(
                    {
                        "[OpenAIModel.generate] Obtained null response": None,
                        "Finish reason": c.finish_reason,
                        "Context": context,
                        "Action": "Outputting an empty string for this completion.",
                    }
                )

        output = np.array(
            [
                "" if c.message.content is None else c.message.content
                for c in output.choices
            ]
        )

        return output, info

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the OpenAI model.")
