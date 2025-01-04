"""WebUI Configuration Module

This module handles configuration for the WebUI, including API providers and model settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ProviderConfig:
    """Configuration for an API provider."""
    env_key: str
    api_url: Optional[str]
    headers: Optional[Dict[str, str]]
    client_library: bool
    default_models: List[str]

class WebUIConfig:
    """WebUI Configuration."""
    
    PROVIDER_CONFIGS = {
        'openai': ProviderConfig(
            env_key='OPENAI_API_KEY',
            api_url=None,
            headers=None,
            client_library=True,
            default_models=['gpt-4', 'gpt-3.5-turbo']
        ),
        'anthropic': ProviderConfig(
            env_key='ANTHROPIC_API_KEY',
            api_url=None,
            headers=None,
            client_library=True,
            default_models=['claude-3-opus-20240229', 'claude-3-sonnet-20240229']
        ),
        'openrouter': ProviderConfig(
            env_key='OPENROUTER_API_KEY',
            api_url='https://openrouter.ai/api/v1',
            headers={
                'HTTP-Referer': 'https://github.com/sosacrazy126/RA.Aid',
                'X-Title': 'RA.Aid'
            },
            client_library=False,
            default_models=['openai/gpt-4-turbo', 'anthropic/claude-3-opus']
        )
    }

    @staticmethod
    def get_configured_providers() -> Dict[str, ProviderConfig]:
        """Get configured providers from environment variables."""
        return {
            name: config 
            for name, config in WebUIConfig.PROVIDER_CONFIGS.items() 
            if os.getenv(config.env_key)
        }

    @staticmethod
    def load_environment_status() -> Dict[str, bool]:
        """Load environment configuration status."""
        return {
            name: bool(os.getenv(config.env_key))
            for name, config in WebUIConfig.PROVIDER_CONFIGS.items()
        }

    @staticmethod
    def get_provider_headers(provider: str) -> Optional[Dict[str, str]]:
        """Get headers for a specific provider."""
        config = WebUIConfig.PROVIDER_CONFIGS.get(provider)
        if not config or not config.headers:
            return None
            
        headers = config.headers.copy()
        if 'Authorization' not in headers and config.env_key:
            api_key = os.getenv(config.env_key)
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
        return headers

    @staticmethod
    def get_provider_url(provider: str) -> Optional[str]:
        """Get API URL for a specific provider."""
        config = WebUIConfig.PROVIDER_CONFIGS.get(provider)
        return config.api_url if config else None
