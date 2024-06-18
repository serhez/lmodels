# LModels

A collection of interfaces to interact with different language models.

## Common API

A common API is used for all available models. This means that, while different specific terms for the same setting may be used in the underlying models (e.g., `temperature = 0.0` vs. `do_sample = False`), this library will expose the same setting with the same name across all models. However, this does not guarantee that all settings will be available for all models nor that the functioning of the setting will be equivalent for all models; refer to the underlying models' API to understand how the available settings affect their functioning. Our API exposes the following common settings:

- `max_tokens`: the maximum number of tokens to generate per context.
- `architecture`: the specific model architecture and pre-trained weights to use; not all models support architecture switching, supporting only the initial loading of the architecture at initialization time.
- `temperature`: the temperature to use during generation, where the value `0.0` will imply greediness in the responses.
    - Note that most model implementations will arrive at a "divide-by-zero" issue when setting `temperature = 0.0`. This is why some of them will not achieve perfectly greedy token selection, as they will replace a zero value by a very small one.
- `top_k`: the number of tokens with the highest probability which will be considered when sampling.
- `top_p`: the cumulative probability threshold for nucleus sampling.
- `n_beams`: the number of beams to use for beam search.
- `return_logprobs`: whether to return the log probabilities of the generated tokens.

Any setting can be provided to a model instance, regardless of the specific type of the model; unsupported settings will just be ignored. This enables you to build codebases without having to manage the differences of each model, making "plug and play" possible. Supported settings are documented in the docstrings of the relevant methods for each model class, as well as in the documentation [TODO: link].

## Mock model

A `MockModel` can be used to debug or quickly iterate during the development of new ideas, whenever the performance of the underlying model is irrelevant for experimentation. You can provide multiple responses matching different inputs complying to pre-defined regular expressions. A probability distribution can also be defined for a group of responses to each matching input. The output can also be random without the need of pre-defined responses, in which case random English words will be produced.

## Developing new models

New models must implement the `Model` class. This guarantees everyone's ability to "plug and play" models for swift iteration, without the need to adhere to new APIs. If you are unhappy with the current `Model` interface, please open an issue with your proposed changes.

> [!NOTE]
>  **Why not just use HuggingFace?** Our interface serves as a wrapper to HuggingFace workflows, and it works in similar ways. However, we aim to also support models not available through HuggingFace, such as OpenAI LLMs.

## Recommended integration with other libraries
This package depends on external representations of data sources. `lmodels` is quite flexible with respect to these representations, and we provide always simple interfaces that can be implemented by the user to wrap their existing code and integrate it with this library. Nonetheless, these interfaces are designed after [`ldata`](https://github.com/serhez/ldata), hence its usage is recommended. `ldata` provides dataset and benchmark abstractions that make training, fine-tuning and evaluation of language-models easier.
