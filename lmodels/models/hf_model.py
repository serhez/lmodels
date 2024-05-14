from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lmodels.utils import DType, Usage, classproperty

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizer,
        pipeline,
    )
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

        dtype: DType = DType.bfloat16
        """
        The data type to use for the model's weights.
        Note that the type of this attribute is the enum `DType` from `lmodels.utils.types`, not a `torch.dtype`.
        - This is necessary for this config to be composable by `hydra`, as it does not support custom classes (e.g., `torch.dtype`) as attribute types.
        - Internally, the `torch.dtype` is inferred from this attribute.
        """

        attention_type: str = "flash_attention_2"
        """The implementation of the attention mechanism to use."""

    @classproperty
    def config_cls(cls) -> type[Config]:
        return cls.Config

    @classproperty
    def generation_info_cls(cls) -> type[Model.GenerationInfo]:
        return cls.GenerationInfo

    def __init__(self, config: Config, logger: Logger | None = None):
        """
        Initializes the Hugging Face model.

        ### Parameters
        ----------
        `config`: the configuration for the Hugging Face model.
        [optional] `logger`: the logger to be used, complying with the `Logger` protocol specified in this library.
        """

        super().__init__(config, logger)
        self._config: HFModel.Config  # pyright is too dumb to infer this

        assert (
            "HF_API_TOKEN" in os.environ
        ), "You must set the `HF_API_TOKEN` environment variable to use the Hugging Face models."
        api_token = os.environ["HF_API_TOKEN"]

        model = AutoModelForCausalLM.from_pretrained(
            config.architecture,
            torch_dtype=config.dtype.torch,
            attn_implementation=config.attention_type,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.architecture, token=api_token, pad_token_id=model.config.eos_token_id
        )
        self._pipeline = pipeline(
            "text-generation",
            token=api_token,
            model=model,
            tokenizer=self._tokenizer,
            torch_dtype=config.dtype.torch,
            device_map="auto",
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def _generate_batch(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: int | None = None,
        unsafe: bool = False,
    ) -> tuple[npt.NDArray[np.str_], Model.GenerationInfo]:
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
            return_text=True,
        )

        if outputs is None:
            response = np.array([[""] * n_samples] * len(inputs))
        elif isinstance(outputs, dict):
            response = np.array(
                [
                    [
                        ""
                        if outputs is None
                        or "generated_text" not in outputs
                        or outputs["generated_text"] is None
                        else outputs["generated_text"][len(input) :],
                    ]
                ]
            )
        elif isinstance(outputs, (list, np.ndarray)):
            response = np.empty((len(inputs), n_samples), dtype=np.str_)
            for i, output in enumerate(outputs):
                if output is None:
                    response[i, :] = ""
                    continue
                for j, sample in enumerate(output):
                    if sample is None or sample["generated_text"] is None:
                        response[i, j] = ""
                    else:
                        response[i, j] = sample["generated_text"][len(inputs[i]) :]
        else:
            raise ValueError(f"[HFModel] Unexpected batch output type: {type(outputs)}")

        info = Model.GenerationInfo(
            usage=Usage(
                n_calls=len(inputs) * n_samples,
                n_tokens_context=sum(
                    [len(self._tokenizer.encode(input)) for input in inputs]
                ),
                n_tokens_output=sum(
                    sum([len(self._tokenizer.encode(o)) for o in output])
                    for output in response
                ),
            )
        )
        self.usage += info.usage

        self._logger.debug(
            {
                "[HFModel.generate]": None,
                "Batch context": context,
                "Batch input": inputs,
                "Batch output": response,
                "N. samples": n_samples,
                "Max. tokens": max_tokens,
                "Info": info,
            }
        )

        return response, info

    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
    ) -> tuple[npt.NDArray[np.str_], Model.GenerationInfo]:
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
            return_text=True,
        )
        if output is None:
            response = np.array([""] * n_samples)
        elif isinstance(output, dict):
            response = np.array(
                [
                    ""
                    if output is None
                    or "generated_text" not in output
                    or output["generated_text"] is None
                    else output["generated_text"][len(input) :],
                ]
            )
        elif isinstance(output, list):
            response = np.array(
                [
                    ""
                    if sample is None
                    or "generated_text" not in sample
                    or sample["generated_text"] is None
                    else sample["generated_text"][len(input) :]
                    for sample in output
                ]
            )
        else:
            raise ValueError(f"[HFModel] Unexpected output type: {type(output)}")

        info = Model.GenerationInfo(
            usage=Usage(
                n_calls=n_samples,
                n_tokens_context=len(self._tokenizer.encode(input)),
                n_tokens_output=sum([len(self._tokenizer.encode(o)) for o in response]),
            )
        )
        self.usage += info.usage

        return response, info

    def fine_tune(self, _):
        raise NotImplementedError(
            "Fine-tuning is not supported for the Hugging Face model."
        )
