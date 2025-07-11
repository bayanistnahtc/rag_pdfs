from llm.base import BaseLLMClient
from llm.mistral import MistralLLMClient
from llm.openai import OpenAILLMClient

class LLMFactory:
    """
    Factory for creating LLM clients.
    """
    _clients = {
        "openai": OpenAILLMClient,
        "mistral": MistralLLMClient
        # TODO: Add support for local models, for example via Ollama.
    }

    @classmethod
    def create_llm(cls, config) -> BaseLLMClient:
        model_type = config.model_type.lower()
        client_class = cls._clients.get(model_type)
        if not client_class:
            raise f"Unsupported LLM type: {model_type}"
        return client_class(config)
