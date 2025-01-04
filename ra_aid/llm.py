import os
from typing import Optional, Union
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from webui.config import WebUIConfig

def initialize_llm(
    provider: Union[str, WebUIConfig],
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> BaseChatModel:
    """Initialize a language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible')
                 or a WebUIConfig object
        model_name: Name of the model to use. If provider is WebUIConfig, this is optional
        temperature: Optional temperature setting for controlling randomness (0.0-2.0).
                    If not specified, provider-specific defaults are used.
        max_tokens: Optional maximum number of tokens to generate.
                   If not specified, provider-specific defaults are used.

    Returns:
        BaseChatModel: Configured language model client

    Raises:
        ValueError: If the provider is not supported
    """
    # Handle WebUIConfig input
    if isinstance(provider, WebUIConfig):
        config = provider
        provider_name = config.provider
        if '/' in config.model:
            # Extract model name from provider/model format
            _, model_name = config.model.split('/')
        else:
            model_name = config.model
        temperature = getattr(config, 'temperature', temperature)
        max_tokens = getattr(config, 'max_tokens', max_tokens)
    else:
        provider_name = provider
        if model_name and '/' in model_name:
            # Extract model name from provider/model format
            _, model_name = model_name.split('/')

    # Common parameters for all providers
    common_params = {}
    if temperature is not None:
        common_params['temperature'] = temperature
    if max_tokens is not None:
        common_params['max_tokens'] = max_tokens

    if provider_name == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            **common_params
        )
    elif provider_name == "anthropic":
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name=model_name,
            **common_params
        )
    elif provider_name == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            **common_params
        )
    elif provider_name == "openai-compatible":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            model=model_name,
            **common_params
        )
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

def initialize_expert_llm(
    provider: Union[str, WebUIConfig] = "openai",
    model_name: Optional[str] = "o1-preview",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> BaseChatModel:
    """Initialize an expert language model client based on the specified provider and model.

    Note: Environment variables must be validated before calling this function.
    Use validate_environment() to ensure all required variables are set.

    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible')
                 or a WebUIConfig object. Defaults to 'openai'.
        model_name: Name of the model to use. Defaults to 'o1-preview'.
        temperature: Optional temperature setting for controlling randomness.
        max_tokens: Optional maximum number of tokens to generate.

    Returns:
        BaseChatModel: Configured expert language model client

    Raises:
        ValueError: If the provider is not supported
    """
    # Handle WebUIConfig input
    if isinstance(provider, WebUIConfig):
        config = provider
        provider_name = config.provider
        model_name = model_name or config.model  # Use provided model_name or config.model
        temperature = getattr(config, 'temperature', temperature)
        max_tokens = getattr(config, 'max_tokens', max_tokens)
    else:
        provider_name = provider

    # Common parameters for all providers
    common_params = {}
    if temperature is not None:
        common_params['temperature'] = temperature
    if max_tokens is not None:
        common_params['max_tokens'] = max_tokens

    if provider_name == "openai":
        return ChatOpenAI(
            api_key=os.getenv("EXPERT_OPENAI_API_KEY"),
            model=model_name,
            **common_params
        )
    elif provider_name == "anthropic":
        return ChatAnthropic(
            api_key=os.getenv("EXPERT_ANTHROPIC_API_KEY"),
            model_name=model_name,
            **common_params
        )
    elif provider_name == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("EXPERT_OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            **common_params
        )
    elif provider_name == "openai-compatible":
        return ChatOpenAI(
            api_key=os.getenv("EXPERT_OPENAI_API_KEY"),
            base_url=os.getenv("EXPERT_OPENAI_API_BASE"),
            model=model_name,
            **common_params
        )
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

def create_model(provider: str, model_name: str, temperature: float = 0.7, max_tokens: int = 2000) -> BaseChatModel:
    """Create a language model instance with the specified configuration.
    
    Args:
        provider: The LLM provider to use ('openai', 'anthropic', 'openrouter', 'openai-compatible')
        model_name: Name of the model to use
        temperature: Temperature setting for controlling randomness (0.0-2.0)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        BaseChatModel: Configured language model client
    """
    return initialize_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
