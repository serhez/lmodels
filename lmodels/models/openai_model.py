from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
import transformers

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    raise ImportError("You must install the `openai` package to use the OpenAI models.")

from lcommon.protocols import Logger
from lcommon.types import AnnotatedConversation
from lcommon.utils import Usage, classproperty

from lmodels import Model


class OpenAIModel(Model):
    """
    An API wrapper for interacting with OpenAI models.

    Your API key must be stored in the environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` (for Azure API).
    If you are using the Azure API, you must also set the `AZURE_OPENAI_ENDPOINT` and `OPENAI_API_VERSION` environment variables.
    Other environment variables that can optionally be set are:
    - `OPENAI_ORG_ID`
    - `AZURE_OPENAI_AD_TOKEN`
    """

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the OpenAI model."""

        name: str = "OpenAIModel"
        """The name of the model."""

        use_azure: bool = False
        """Whether to use the Azure OpenAI API instead of the regular one."""

        url_replacements: dict[str, str] = field(default_factory=lambda: dict())
        """
        A dictionary of URL replacements to be made for the Azure API url.
        The keys are the patterns to be identified in the `URL.path`, and the values are the whole new `URL.path` to be used instead.
        """

        architecture: str
        """
        The default name of the model architecture to use.
        Must be listed as an OpenAI model architecture.
        """

        temperature: float = 1.0
        """The default temperature for sampling from the model."""

        top_p: float = 1.0
        """The default cumulative probability for nucleus sampling."""

    @dataclass(kw_only=True)
    class GenerationInfo(Model.GenerationInfo):
        """The generation information for the OpenAI model."""

        finish_reasons: list[list[str | None]] = field(default_factory=lambda: list())
        """
        The reasons for finishing the generation, for each output and sample.
        If the generation was not finished (e.g., an exception was thrown), the value is `None`.
        - In such case, only a single `None` is appended to the inner list, hence the length of the inner list will not be equal to the number of requested samples.
        """

        def __add__(
            self, other: OpenAIModel.GenerationInfo | None
        ) -> OpenAIModel.GenerationInfo:
            return OpenAIModel.GenerationInfo(
                usage=self.usage + other.usage if other is not None else self.usage,
                finish_reasons=self.finish_reasons + other.finish_reasons
                if other is not None
                else self.finish_reasons + [[None]],
            )

        def __radd__(
            self, other: OpenAIModel.GenerationInfo | None
        ) -> OpenAIModel.GenerationInfo:
            return self + other

        def __iadd__(
            self, other: OpenAIModel.GenerationInfo | None
        ) -> OpenAIModel.GenerationInfo:
            if other is not None:
                self.usage += other.usage
                self.finish_reasons += other.finish_reasons
            else:
                self.finish_reasons += [[None]]  # type: ignore[reportAttributeAccessIssue]
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
                "finish_reasons": self.finish_reasons,
            }

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

    def generate(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
        architecture: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
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
        - If `None`, the default number of samples specified in the model's configuration is used.
        `architecture`: the name of the model architecture to use.
        - If `None`, the default architecture specified in the model's configuration is used.
        `temperature`: the temperature for sampling from the model.
        - If `None`, the default temperature specified in the model's configuration is used.
        `top_p`: the cumulative probability for nucleus sampling.
        - If `None`, the default cumulative probability specified in the model's configuration is used.

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

        return super().generate(  # type: ignore[reportReturnType]
            context,
            n_samples,
            max_tokens,
            architecture=architecture,
            temperature=temperature,
            top_p=top_p,
        )

    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
        architecture: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.str_], GenerationInfo]:
        """
        The model's internal implementation of `generate` acting on a single annotated conversation (i.e., list of dict messages).

        ### Parameters
        ----------
        `context`: the context to generate from.
        - Each message (dictionary) must contain the `content` field.
        `n_samples`: the number of samples to generate for the context string.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If `None`, `config.max_tokens` will be used.
        `architecture`: the name of the model architecture to use.
        - If `None`, `config.architecture` will be used.
        `temperature`: the temperature for sampling from the model.
        - If `None`, `config.temperature` will be used.
        `top_p`: the cumulative probability for nucleus sampling.
        - If `None`, `config.top_p` will be used.

        ### Returns
        -------
        A tuple containing:
        - A `numpy.NDArray` with the generated tokens for each sample of shape (`n_samples`).
        - Extra information about the generation process of the model.
        """

        if max_tokens is None:
            max_tokens = self._config.max_tokens
        if architecture is None:
            architecture = self._config.architecture
        if temperature is None:
            temperature = self._config.temperature
        if top_p is None:
            top_p = self._config.top_p

        output = self._client.chat.completions.create(
            messages=context,  # type: ignore[reportArgumentType]
            model=architecture,
            max_tokens=max_tokens,
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
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
