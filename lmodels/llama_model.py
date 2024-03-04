import json
import os
import sys
from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch
from mloggers import Logger

try:
    from fairscale.nn.model_parallel.initialize import (
        get_model_parallel_rank,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from llama import Llama
    from llama.model import ModelArgs, Transformer
    from llama.tokenizer import Tokenizer
except ImportError:
    raise ImportError(
        "You must install the [`llama`](https://github.com/facebookresearch/llama) and `fairscale` packages to use the Llama models."
    )


from lmodels.model import AnnotatedConversation, Context, Model


class LlamaModel(Model):
    """
    An API wrapper for interacting with Llama models through the Llama API.
    """

    @dataclass(kw_only=True)
    class Config(Model.Config):
        """The configuration for the Llama model."""

        name: str = "LlamaModel"
        """The name of the model."""

        checkpoint_dir: str = MISSING
        """The path to the model's checkpoint directory."""

        tokenizer_path: str = MISSING
        """The path to the model's tokenizer."""

        top_p: float = 0.9
        """The nucleus sampling probability."""

        temperature: float = 0.6
        """The sampling temperature."""

        max_batch_size: int = 8
        """The maximum batch size for the model."""

        model_parallel_size: Optional[int] = None
        """
        The number of model parallel GPUs to use.
        If None, the value is taken from the environment variable `WORLD_SIZE`.
        If the environment variable is not set, the value is set to 1.
        """

    def __init__(self, config: Config, logger: Optional[Logger] = None):
        """
        Initializes the Llama model.

        ### Parameters
        ----------
        `config`: the configuration for the Llama model.
        [optional] `logger`: the logger to be used.
        """

        super().__init__(config, logger)

        torch.set_default_tensor_type(torch.cuda.HalfTensor)

        # Manage the device for multi-GPU training
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if config.model_parallel_size is None:
                config.model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(config.model_parallel_size)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        # Load the checkpoints
        checkpoints = sorted(Path(config.checkpoint_dir).glob("*.pth"))
        assert (
            len(checkpoints) > 0
        ), f"no checkpoint files found in {config.checkpoint_dir}"
        assert (
            config.model_parallel_size == len(checkpoints)
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {config.model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Create the tokenizer
        self._tokenizer = Tokenizer(model_path=config.tokenizer_path)

        # Create the model
        try:
            with open(Path(config.checkpoint_dir) / "params.json", "r") as f:
                params = json.loads(f.read())
        except FileNotFoundError:
            params = {}
        model_args: ModelArgs = ModelArgs(
            max_seq_len=config.default_max_tokens,
            max_batch_size=config.max_batch_size,
            **params,
        )
        model_args.vocab_size = self._tokenizer.n_words
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        # Create the generator
        self._generator = Llama(model, self._tokenizer)

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def generate(
        self,
        context: Context,
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
        unsafe: bool = False,
    ) -> npt.NDArray[np.str_]:
        context = self._parse_context(context, unsafe=unsafe)
        if len(context) == 1:
            return np.array([self._generate_impl(context[0], n_samples, max_tokens)])

        inputs = []
        for conversation in context:
            input = conversation[0]["content"]
            for i in range(1, len(conversation)):
                input += "\n" + conversation[i]["content"]
            inputs.append(input)
        input_tokens = [
            self._tokenizer.encode(input, bos=True, eos=False) for input in inputs
        ]

        if max_tokens is None:
            max_tokens = self._config.default_max_tokens

        outputs = []
        for _ in range(n_samples):
            outputs = self._generator.generate(
                prompt_tokens=input_tokens,
                max_gen_len=max_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                logprobs=False,
                echo=False,
            )
            outputs.append([self._tokenizer.decode(output) for output in outputs])
        outputs = np.array(outputs).T

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[LlamaModel.generate]": None,
                    "Batch context": context,
                    "Batch input": inputs,
                    "Batch output": outputs,
                    "n_samples": n_samples,
                    "max_tokens": max_tokens,
                }
            )

        return outputs

    def _generate_impl(
        self,
        context: AnnotatedConversation,
        n_samples: int = 1,
        max_tokens: Optional[int] = None,
    ) -> npt.NDArray[np.str_]:
        if max_tokens is None:
            max_tokens = self._config.default_max_tokens

        input = context[0]["content"]
        for i in range(1, len(context)):
            input += "\n" + context[i]["content"]
        input_tokens = [self._tokenizer.encode(input, bos=True, eos=False)]
        input_tokens = input_tokens * n_samples

        output = self._generator.generate(
            prompt_tokens=input_tokens,
            max_gen_len=max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            logprobs=False,
            echo=False,
        )
        output = self._tokenizer.decode(output[0])

        if self._logger and self._config.debug:
            self._logger.debug(
                {
                    "[LlamaModel.generate]": None,
                    "Context": context,
                    "Input": input,
                    "Output": output,
                    "n_samples": n_samples,
                    "max_tokens": max_tokens,
                }
            )

        return output

    def fine_tune(self, _):
        raise NotImplementedError("Fine-tuning is not supported for the Llama model.")
