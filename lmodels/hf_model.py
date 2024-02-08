from dataclasses import dataclass

import torch
import transformers
from omegaconf import MISSING

from lmodels.model import Model


class HFModel(Model):
    """
    An API wrapper for interacting with Hugging Face models.
    """

    @dataclass(kw_only=True)
    class Config(Model.Config):
        api_token: str = MISSING
        """The API token to use for the model."""

        model_name: str = MISSING
        """The name of the model to use. Must be listed as a Hugging Face model."""

        do_sample: bool = True
        """Whether to sample from the model's output."""

        top_k: int = 10
        """The number of top tokens to consider when sampling."""

        num_return_sequences: int = 1
        """The number of sequences to return when sampling."""

    def __init__(self, config: Config):
        """
        Initializes the Hugging Face model.

        ### Parameters
        ----------
        `config`: the configuration for the Hugging Face model.
        """

        super().__init__(config)

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_name, token=config.api_token
        )
        self._pipeline = transformers.pipeline(
            "text-generation",
            token=config.api_token,
            model=config.model_name,
            torch_dtype=torch.float16,
            device_map=config.device,
        )
        self._do_sample = config.do_sample
        self._top_k = config.top_k
        self._num_return_sequences = config.num_return_sequences

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return self._tokenizer

    def _generate_impl(self, context: str, max_tokens: int = 1) -> str:
        generated_tokens = self._pipeline(
            context,
            do_sample=self._do_sample,
            top_k=self._top_k,
            num_return_sequences=self._num_return_sequences,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )[0]["generated_text"][len(context) :]

        return generated_tokens

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the mock model.")
