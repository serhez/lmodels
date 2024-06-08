from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from lcommon.types import AnnotatedConversation, Context, DType
from lcommon.utils import Usage, classproperty, merge_system_messages

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedTokenizer,
    )
    # FIX:
    # from transformers import __version__ as transformers_version
    #
    # assert (
    #     tuple(map(int, transformers_version.split("."))) >= (4, 41)
    # ), "You must install the `transformers[torch]` package with version >= 4.41 to use the Hugging Face models."
except ImportError:
    raise ImportError(
        "You must install the `transformers[torch] >= 4.41` package to use the Hugging Face models."
    )

from lcommon.protocols import Logger

from lmodels.model import Model


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

        device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        """The device to use for the model. It is chosen automatically based on the available hardware."""

        dtype: DType = DType.bfloat16
        """
        The data type to use for the model's weights.
        Note that the type of this attribute is the enum `DType` from `lmodels.utils.types`, not a `torch.dtype`.
        - This is necessary for this config to be composable by `hydra`, as it does not support custom classes (e.g., `torch.dtype`) as attribute types.
        - Internally, the `torch.dtype` is inferred from this attribute.
        """

        attention_type: str = "flash_attention_2"
        """The implementation of the attention mechanism to use."""

        padding_side: str = "left"
        """The side to pad the input sequences."""

        architecture: str
        """The name of the architecture to use. Must be listed as a Hugging Face architecture."""

        temperature: float = 0.1
        """
        The default temperature to use when sampling from the model's output.
        It must be greater than 0.0. For deterministic outputs, set `Config.do_sample = False`.
        """

        do_sample: bool = True
        """The default value of whether to sample from the model's output."""

        top_k: int = 10
        """The default number of top tokens to consider when sampling."""

        top_p: float = 0.9
        """The default cumulative probability threshold for nucleus sampling."""

        n_beams: int = 1
        """The default number of beams to use for beam search."""

        load_from_cache: bool = False
        """
        Whether to load the tokenizer and model weights from a cache directory.
        The cache directory is specified by the `HF_HOME` environment variable, or the current directory if not set.
        """

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

        if not config.load_from_cache:
            cache_dir = None
        elif "HF_HOME" in os.environ:
            cache_dir = os.environ["HF_HOME"]
        else:
            cache_dir = "."
        if cache_dir is not None:
            self._logger.debug(f"[HFModel] Using cache directory: {cache_dir}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.architecture,
            token=api_token,
            padding_side=config.padding_side,
            cache_dir=cache_dir,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = config.padding_side

        self._model = AutoModelForCausalLM.from_pretrained(
            config.architecture,
            token=api_token,
            torch_dtype=config.dtype.torch,
            attn_implementation=config.attention_type,
            device_map="auto",
            cache_dir=cache_dir,
        )
        self._model.eval()

        self._fix_init_models()

    def _fix_init_models(self):
        """A collection of architecture-specific fixes required at initialization for the Hugging Face models."""

        # WizardCoder extra token fix
        # https://github.com/huggingface/transformers/issues/24843
        if "WizardLM" in self._config.architecture:
            special_token_dict = self._tokenizer.special_tokens_map
            self._tokenizer.add_special_tokens(special_token_dict)
            self._model.resize_token_embeddings(len(self._tokenizer))
            self._logger.debug(
                f"[HFModel] Special tokens added to tokenizer: {special_token_dict}"
            )

    @property
    def _should_merge_system(self):
        """
        Whether to merge system messages in the input.
        This is necessary for some models that do not support system messages.
        """

        return True if "mistralai" in self._config.architecture.lower() else False

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def generate(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: str | None = None,
        top_k: float | None = None,
        top_p: float | None = None,
        n_beams: int | None = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.str_], Model.GenerationInfo]:
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
        - If `None`, the default value specified in the model's configuration is used.
        `n_samples`: the number of samples to generate for each context string.
        - If `None`, the default number of samples specified in the model's configuration is used.
        `temperature`: the temperature to use when sampling from the model's output.
        - It must be greater than 0.0. For deterministic outputs, set `do_sample = False`.
        - If `None`, the default value specified in the model's configuration is used.
        `do_sample`: whether to sample from the model's output.
        - If `None`, the default value specified in the model's configuration is used.
        `top_k`: the number of top tokens to consider when sampling.
        - If `None`, the default value specified in the model's configuration is used.
        `top_p`: the cumulative probability threshold for nucleus sampling.
        - If `None`, the default value specified in the model's configuration is used.
        `n_beams`: the number of beams to use for beam search.
        - If `None`, the default value specified in the model's configuration is used.

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

        return super().generate(
            context,
            n_samples,
            max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            n_beams=n_beams,
        )

    def _generate_batch(
        self,
        context: list[AnnotatedConversation],
        n_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        n_beams: int | None = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.str_], Model.GenerationInfo]:
        """
        Internal method for generating samples in batches.

        ### Parameters
        ----------
        `context`: the context to generate from.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If `None`, the default value specified in the model's configuration is used.
        `n_samples`: the number of samples to generate for each context string.
        - If `None`, the default number of samples specified in the model's configuration is used.
        `temperature`: the temperature to use when sampling from the model's output.
        - If `None`, the default value specified in the model's configuration is used.
        `do_sample`: whether to sample from the model's output.
        - If `None`, the default value specified in the model's configuration is used.
        `top_k`: the number of top tokens to consider when sampling.
        - If `None`, the default value specified in the model's configuration is used.
        `top_p`: the cumulative probability threshold for nucleus sampling.
        - If `None`, the default value specified in the model's configuration is used.
        `n_beams`: the number of beams to use for beam search.
        - If `None`, the default value specified in the model's configuration is used.

        ### Returns
        -------
        A tuple containing:
        - A `numpy.NDArray` of strings of shape (`len(context)`, `n_samples`).
            - If `context` is a single string/dictionary, then `len(context)` is 1.
        - Extra information about the generation process of the model.

        ### Raises
        -------
        `ValueError`: if the input type is not supported.
        `ValueError`: if a message dictionary does not contain a `content` field.
        `AssertionError`: if a context list is provided and it is empty.
        """

        if max_tokens is None:
            max_tokens = self._config.max_tokens
        if temperature is None:
            temperature = self._config.temperature
        if do_sample is None:
            do_sample = self._config.do_sample
        if top_k is None:
            top_k = self._config.top_k
        if top_p is None:
            top_p = self._config.top_p
        if n_beams is None:
            n_beams = self._config.n_beams

        if np.isclose(temperature, 0.0):
            do_sample = False
            temperature = None
            top_k = None
            top_p = None

        if self._should_merge_system:
            context = merge_system_messages(context)

        # Apply an architecture-specific context template
        input_tkns = self._tokenizer.apply_chat_template(
            context,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._config.device)

        # Generate the output tokens
        # The outputs are all appended to a tensor of length len(inputs) * n_samples
        # We want to reshape it to a nested tensor of dimensions (len(inputs), n_samples)
        output_tkns = self._model.generate(
            **input_tkns,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=n_samples,
            num_beams=n_beams,
            pad_token_id=self._tokenizer.eos_token_id,
        ).reshape(len(context), n_samples, -1)

        # Decode the tokens
        inputs = np.array(
            [
                self._tokenizer.decode(
                    tkns,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                for tkns in input_tkns["input_ids"]
            ]
        )
        outputs = np.array(
            [
                [
                    self._tokenizer.decode(
                        tkns[len(input_tkns["input_ids"][i]) :],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    for tkns in output_tkns[i]  # each sample
                ]
                for i in range(len(output_tkns))  # each input
            ]
        )

        # Record the generation information
        info = Model.GenerationInfo(
            usage=Usage(
                n_calls=len(context),
                n_tokens_context=sum(
                    len(input_tkns["input_ids"][i]) for i in range(len(context))
                ),
                n_tokens_output=sum(
                    sum(
                        len(output_tkns[i][j]) - len(input_tkns["input_ids"][i])
                        for j in range(n_samples)
                    )
                    for i in range(len(context))
                ),
            )
        )
        self.usage += info.usage

        self._logger.debug(
            {
                "[HFModel.generate]": None,
                "Context": context,
                "Inputs": inputs,
                "Outputs": outputs,
                "N. samples": n_samples,
                "Max. tokens": max_tokens,
                "Temperature": temperature,
                "Do sample": do_sample,
                "Top k": top_k,
                "Top p": top_p,
                "N. beams": n_beams,
                "Info": info,
            }
        )

        return outputs, info

    def _generate_single(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        n_beams: int | None = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.str_], Model.GenerationInfo]:
        """
        The model's internal implementation of `generate` acting on a single conversation (i.e., list of messages).

        ### Parameters
        ----------
        `context`: the context to generate from.
        - Each message (dictionary) must contain the `content` field.
        `n_samples`: the number of samples to generate for the context string.
        `max_tokens`: the maximum number of tokens to generate per context string.
        - If `None`, `config.max_tokens` will be used.
        `temperature`: the temperature to use when sampling from the model's output.
        - If `None`, `config.temperature` will be used.
        `do_sample`: whether to sample from the model's output.
        - If `None`, `config.do_sample` will be used.
        `top_k`: the number of top tokens to consider when sampling.
        - If `None`, `config.top_k` will be used.
        `top_p`: the cumulative probability threshold for nucleus sampling.
        - If `None`, the default value specified in the model's configuration is used.
        `n_beams`: the number of beams to use for beam search.
        - If `None`, the default value specified in the model's configuration is used.

        ### Returns
        -------
        A tuple containing:
        - A `numpy.NDArray` with the generated tokens for each sample of shape (`n_samples`).
        - Extra information about the generation process of the model.
        """

        outputs, info = self._generate_batch(
            [context],
            n_samples,
            max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            n_beams=n_beams,
        )

        return outputs[0], info

    def fine_tune(self, _):
        raise NotImplementedError(
            "Fine-tuning is not supported for the Hugging Face model."
        )
