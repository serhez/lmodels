# LModels

A collection of interfaces to interact with different language models, by Aalto LaReL research group.

## Developing new models

New models must implement the `Model` class. This guarantees everyone's ability to "plug and play" models to quickly iterate and test ideas, without the need to adhere to new APIs. If you are unhappy with the current `Model` interface, please open an issue with your proposed changes.

> [!NOTE]
>  **Why not just use HuggingFace?** Our interface serves as a wrapper to HuggingFace `pipeline` workflows, and it works in similar ways. However, we aim to also support models not available through HuggingFace, such as OpenAI LLMs.

## Integration with other libraries
This package depends on external representations of data sources. `lmodels` is quite flexible with respect to these representations, and we provide always simple interfaces that can be implemented by the user to wrap their existing code and integrate it with this library. Nonetheless, these interfaces are designed after [`ldata`](https://github.com/serhez/ldata), hence it's usage is recommended. `ldata` provides dataset and benchmark abstractions that make training, fine-tuning and evaluation of language-models easier.

> [!NOTE]
> While you do not have to use `ldata` objects to use any part of our library, it currently is listed as a dependency of this package because we use their types. In the future, we plan on dropping this dependency to make this package more self-contained and light-weight.
