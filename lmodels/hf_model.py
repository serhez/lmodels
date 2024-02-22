from dataclasses import MISSING, dataclass
from typing import Iterator, List, Optional, Union

import torch
import transformers
from ldata import Dataset
from mloggers import Logger

from lmodels.model import Model


class HFModel(Model):
    """
    An API wrapper for interacting with Hugging Face models.
    """

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the Hugging Face model."""

        name: str = "HFModel"
        """The name of the model."""

        api_token: str = MISSING
        """The API token to use for the model."""

        architecture: str = MISSING
        """The name of the architecture to use. Must be listed as a Hugging Face architecture."""

        do_sample: bool = True
        """Whether to sample from the model's output."""

        top_k: int = 10
        """The number of top tokens to consider when sampling."""

        num_return_sequences: int = 1
        """The number of sequences to return when sampling."""

    def __init__(self, config: Config, logger: Optional[Logger] = None):
        """
        Initializes the Hugging Face model.

        ### Parameters
        ----------
        `config`: the configuration for the Hugging Face model.
        [optional] `logger`: the logger to be used.
        """

        super().__init__(config, logger)

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.architecture, token=config.api_token
        )
        self._pipeline = transformers.pipeline(
            "text-generation",
            token=config.api_token,
            model=config.architecture,
            torch_dtype=torch.float16,
            device_map=config.device,
        )

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return self._tokenizer

    def generate(
        self,
        context: Union[
            str,
            List[str],
            Iterator[str],
            Dataset[str, str],
        ],
        max_tokens: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """
        Generates the next given number of tokens in the sequence.
        It has similar functionality to HuggingFace's `pipeline` method.

        ### Parameters
        ----------
        `context`: the context/s to generate from.
        - If it is a `Dataset`, the model will generate from all samples in the test set.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If None, the model will generate tokens until the EOS token is produced.

        ### Returns
        -------
        The generated tokens.
        - If `context` is a string, the return value is a string.
        - If `context` is a list or iterator of strings or a `Dataset`, the return value is a list of strings.
        """

        if isinstance(context, str):
            return self._generate_impl(context, max_tokens)
        elif isinstance(context, Dataset):
            context = list(context.test_set.inputs)
        elif isinstance(context, Iterator):
            context = list(context)
        elif not isinstance(context, list):
            raise ValueError(
                f"Invalid type for `context`: {type(context)}. Must be a string, list of strings, iterator returning strings or `Dataset`."
            )

        outputs = self._pipeline(
            context,
            batch_size=self._config.generate_batch_size,
            do_sample=self._config.do_sample,
            top_k=self._config.top_k,
            num_return_sequences=self._config.num_return_sequences,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )[0]["generated_text"][len(context) :]

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[HFModel.generate]": None,
                    "Batch context": context,
                    "Batch output": outputs,
                }
            )

        return outputs

    def _generate_impl(self, context: str, max_tokens: Optional[int] = None) -> str:
        output = self._pipeline(
            context,
            do_sample=self._config.do_sample,
            top_k=self._config.top_k,
            num_return_sequences=self._config.num_return_sequences,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )[0]["generated_text"][len(context) :]

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[HFModel.generate]": None,
                    "Context": context,
                    "Output": output,
                }
            )

        return output

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the mock model.")


try:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="base_hf_model", node=HFModel.Config)
except ModuleNotFoundError:
    pass
