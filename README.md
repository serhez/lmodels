# LModels

A collection of interfaces to interact with different language models, by Aalto LaReL research group.

## Developing new models

New models must implement the `Model` class. This guarantees everyone's ability to "plug and play" models to quickly iterate and test ideas, without the need to adhere to new APIs. If you are unhappy with the current `Model` interface, please open an issue with your proposed changes.

> [!NOTE] > **Why not just use HuggingFace?** Our interface serves as a wrapper to HuggingFace `pipeline` workflows, and it works in similar ways. However, we aim to also support models not available through HuggingFace, such as OpenAI LLMs.
