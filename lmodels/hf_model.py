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
        max_tokens: int = 500,
        n_samples: int = 1,
    ) -> List[List[str]]:
        """
        Generates the next given number of tokens in the sequence.
        It has similar functionality to HuggingFace's `pipeline` method.

        ### Parameters
        ----------
        `context`: the context/s to generate from.
        - If it is a `Dataset`, the model will generate from all samples in the test set.
        `max_tokens`: the maximum number of tokens to generate per context string.
        `n_samples`: the number of samples to generate for each context string.
        - You should consider setting `Config.do_sample = True` if you want `n_samples > 1`.

        ### Returns
        -------
        The generated tokens.
        - The return type is a list of lists of strings of size [`len(context)`, `n_samples`]; if `context` is a string, then `len(context)` is 1.
        """

        if isinstance(context, str):
            return self._generate_impl(context, max_tokens, n_samples)
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
            num_return_sequences=n_samples,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )
        outputs = [
            [sample["generated_text"][len(context) :] for sample in output]
            for output in outputs
        ]

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[HFModel.generate]": None,
                    "Batch context": context,
                    "Batch output": outputs,
                }
            )

        return outputs

    def _generate_impl(
        self, context: str, max_tokens: int = 500, n_samples: int = 1
    ) -> str:
        output = self._pipeline(
            context,
            do_sample=self._config.do_sample,
            top_k=self._config.top_k,
            num_return_sequences=n_samples,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )
        output = [sample["generated_text"][len(context) :] for sample in output]

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
