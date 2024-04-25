import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

try:
    import transformers
except ImportError:
    raise ImportError(
        "You must install the `transformers[torch]` package to use the Hugging Face models."
    )

from lmodels.model import AnnotatedConversation, Context, Model
from lmodels.protocols import Logger


class HFModel(Model):
    """
    An API wrapper for interacting with Hugging Face models.
    Your API token must be stored in the environment variable `HF_API_TOKEN`.
    """

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the Hugging Face model."""

        name: str = "HFModel"
        """The name of the model."""

        architecture: str
        """The name of the architecture to use. Must be listed as a Hugging Face architecture."""

        do_sample: bool = True
        """Whether to sample from the model's output."""

        top_k: int = 10
        """The number of top tokens to consider when sampling."""

    @property
    def config_cls(self) -> type[Config]:
        return HFModel.Config

    def __init__(self, config: Config, logger: Logger | None = None):
        """
        Initializes the Hugging Face model.

        ### Parameters
        ----------
        `config`: the configuration for the Hugging Face model.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        super().__init__(config, logger)

        assert (
            "HF_API_TOKEN" in os.environ
        ), "You must set the `HF_API_TOKEN` environment variable to use the Hugging Face models."
        api_token = os.environ["HF_API_TOKEN"]

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.architecture, token=api_token
        )
        self._pipeline = transformers.pipeline(
            "text-generation",
            token=api_token,
            model=config.architecture,
            torch_dtype=torch.float16,
            device_map=config.device,
        )

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return self._tokenizer

    def _generate_batch(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: int | None = None,
        unsafe: bool = False,
    ) -> tuple[npt.NDArray[np.str_], dict[str, Any]]:
        context = self._parse_context(context, unsafe=unsafe)
        if len(context) == 1:
            return self._generate_single(context[0], n_samples, max_tokens)

        inputs = []
        for conversation in context:
            input = conversation[0]["content"]
            for i in range(1, len(conversation)):
                input += "\n" + conversation[i]["content"]
            inputs.append(input)

        if max_tokens is None:
            max_tokens = self._config.default_max_tokens

        outputs = self._pipeline(
            inputs,
            batch_size=self._config.generate_batch_size,
            do_sample=self._config.do_sample,
            top_k=self._config.top_k,
            num_return_sequences=n_samples,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )

        outputs = np.empty((len(inputs), n_samples), dtype=np.str_)
        for i, output in enumerate(outputs):
            for j, sample in enumerate(output):
                outputs[i, j] = sample["generated_text"][len(inputs[i]) :]

        stats = {
            "n_tokens_context": sum(
                [len(self._tokenizer.encode(input)) for input in inputs]
            ),
            "n_tokens_output": sum(
                sum([len(self._tokenizer.encode(o)) for o in output])
                for output in outputs
            ),
            "n_calls": len(inputs),
        }
        self._record_model_usage(stats)

        self._logger.debug(
            {
                "[HFModel.generate]": None,
                "Batch context": context,
                "Batch input": inputs,
                "Batch output": outputs,
                "N. samples": n_samples,
                "Max. tokens": max_tokens,
                "Usage stats.": stats,
            }
        )

        return outputs, stats

    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
    ) -> tuple[npt.NDArray[np.str_], dict[str, Any]]:
        if max_tokens is None:
            max_tokens = self._config.default_max_tokens

        input = context[0]["content"]
        for i in range(1, len(context)):
            input += "\n" + context[i]["content"]

        output = self._pipeline(
            input,
            do_sample=self._config.do_sample,
            top_k=self._config.top_k,
            num_return_sequences=n_samples,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )
        output = np.array([sample["generated_text"][len(input) :] for sample in output])

        stats = {
            "n_tokens_context": len(self._tokenizer.encode(input)),
            "n_tokens_output": sum([len(self._tokenizer.encode(o)) for o in output]),
            "n_calls": n_samples,
        }
        self._record_model_usage(stats)

        return output, stats

    def fine_tune(self, _):
        raise NotImplementedError(
            "Fine-tuning is not supported for the Hugging Face model."
        )
