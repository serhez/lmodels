"""
A collection of interfaces to interact with different language models.
"""

from .model import Model
from .models.hf_model import HFModel
from .models.mock_model import MockModel, MockResponse, MockResponseCollection
from .models.openai_model import OpenAIModel

__all__ = [
    "HFModel",
    "MockModel",
    "Model",
    "OpenAIModel",
    "MockResponse",
    "MockResponseCollection",
]
