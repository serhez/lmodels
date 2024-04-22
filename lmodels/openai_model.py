import os
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
import transformers
from mloggers import Logger

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    raise ImportError("You must install the `openai` package to use the OpenAI models.")

from lmodels.model import AnnotatedConversation, Model


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

    def __init__(self, config: Config, logger: Logger | None = None):
        """
        Initializes the OpenAI model.
        Your API key should be stored in the environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` (for Azure API).
        If you are using the Azure API, you must also set the `AZURE_OPENAI_ENDPOINT` and `OPENAI_API_VERSION` environment variables.
        Other environment variables that can optionally be set are:
        - `OPENAI_ORG_ID`
        - `AZURE_OPENAI_AD_TOKEN`

        ### Parameters
        ----------
        `config`: the configuration for the OpenAI model.
        [optional] `logger`: the logger to be used.
        """

        super().__init__(config, logger)

        self._config = config

        def _update_base_url(request: httpx.Request) -> None:
            init_url = request.url
            for key, value in config.url_replacements.items():
                if key in request.url.path:
                    request.url = request.url.copy_with(path=value)
            if self._logger and self._config.debug:
                self._logger.debug(
                    {
                        "[OpenAIModel._update_base_url]": None,
                        "Initial URL": str(init_url),
                        "New URL": str(request.url),
                    }
                )

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

        self._stats = {
            "n_tokens_context": 0,
            "n_tokens_output": 0,
            "n_calls": 0,
        }

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        raise NotImplementedError(
            "The OpenAI model does not currently provide a tokenizer."
        )

    @property
    def usage(self) -> dict[str, Any]:
        return self._stats

    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
    ) -> tuple[npt.NDArray[np.str_], dict[str, Any]]:
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
        try:
            print(output.choices[0].finish_reason)
            print(output.usage)
        except Exception as e:
            print(f"No finish reason: {e}")
        output = np.array(
            [
                "" if c.message.content is None else c.message.content
                for c in output.choices
            ]
        )

        stats = {
            "n_tokens_context": output.usage.prompt_tokens,
            "n_tokens_output": output.usage.completion_tokens,
            "n_calls": 1,
        }
        self._stats["n_tokens_context"] += stats["n_tokens_context"]
        self._stats["n_tokens_output"] += stats["n_tokens_output"]
        self._stats["n_calls"] += stats["n_calls"]

        return output, stats

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the OpenAI model.")
