import os
from dataclasses import MISSING, dataclass
from typing import Dict, List, Optional

# import httpx
import numpy as np
import numpy.typing as npt
import transformers
from mloggers import Logger

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    raise ImportError("You must install the `openai` package to use the OpenAI models.")

from lmodels.model import Model


class OpenAIModel(Model):
    """
    An API wrapper for interacting with OpenAI models.

    Your API key must be stored in the environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` (for Azure API).
    If you are using the Azure API, you must also set the `AZURE_OPENAI_ENDPOINT` environment variable.
    Other environment variables that can optionally be set are:
    - `OPENAI_ORG_ID`
    - `AZURE_OPENAI_AD_TOKEN`
    - `OPENAI_API_VERSION`
    """

    _DEFAULT_ROLE = "user"

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the Hugging Face model."""

        name: str = "OpenAIModel"
        """The name of the model."""

        use_azure: bool = False
        """Whether to use the Azure OpenAI API instead of the regular one."""

        architecture: str = MISSING
        """
        The name of the model architecture to use.
        Must be listed as an OpenAI model architecture.
        """

        temperature: float = 1.0
        """The temperature for sampling from the model."""

        top_p: float = 1.0
        """The cumulative probability for nucleus sampling."""

    def __init__(self, config: Config, logger: Optional[Logger] = None):
        """
        Initializes the Hugging Face model.
        Your API key should be stored in the environment variable `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` (for Azure API).

        ### Parameters
        ----------
        `config`: the configuration for the OpenAI model.
        [optional] `logger`: the logger to be used.
        """

        super().__init__(config, logger)

        self._config = config

        # def _update_base_url(request: httpx.Request) -> None:
        #     if request.url.path == "/chat/completions":
        #         request.url = request.url.copy_with(path="/v1/chat")

        if config.use_azure:
            assert (
                not config.use_azure or "AZURE_OPENAI_API_KEY" in os.environ
            ), "you must set the `AZURE_OPENAI_API_KEY` environment variable for `config.use_azure = True`"
            assert (
                not config.use_azure or "AZURE_OPENAI_ENDPOINT" in os.environ
            ), "you must set the `AZURE_OPENAI_ENDPOINT` environment variable for `config.use_azure = True`"

            self._client = AzureOpenAI(
                # http_client=httpx.Client(
                #     event_hooks={
                #         "request": [_update_base_url],
                #     }
                # ),
            )
        else:
            assert (
                config.use_azure or "OPENAI_API_KEY" in os.environ
            ), "you must set the `OPENAI_API_KEY` environment variable for `config.use_azure = False`"

            self._client = OpenAI()

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        raise NotImplementedError(
            "The OpenAI model does not currently provide a tokenizer."
        )

    def _generate_impl(
        self,
        context: List[Dict[str, str]],
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
    ) -> npt.NDArray[np.str_]:
        if max_tokens is None:
            max_tokens = self._config.max_tokens

        for message in context:
            if "role" not in message:
                message["role"] = self._DEFAULT_ROLE

        output = self._client.chat.completions.create(
            messages=context,
            model=self._config.architecture,
            max_tokens=max_tokens,
            n=n_samples,
            temp=self._config.temperature,
            top_p=self._config.top_p,
        )

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[OpenAIModel.generate]": None,
                    "Context": context,
                    "Output": output,
                    "Default role": self._DEFAULT_ROLE,
                    "n_samples": n_samples,
                    "max_tokens": max_tokens,
                }
            )

        return output

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the OpenAI model.")
