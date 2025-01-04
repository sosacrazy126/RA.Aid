import pytest
from unittest.mock import patch, MagicMock
import os
from webui.app import handle_conversation, load_available_models
from webui.config import WebUIConfig
from litellm import completion

@pytest.mark.asyncio
async def test_openrouter_configuration():
    """Test that OpenRouter is configured correctly with litellm."""
    config = WebUIConfig(
        provider="openrouter",
        model="mistralai/mistral-medium",
        research_only=True
    )
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    
    with patch('webui.app.completion') as mock_completion, \
         patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
        
        mock_completion.return_value = mock_response
        
        await handle_conversation("Test message", config)
        
        # Verify completion was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        
        # Check model name format
        assert call_args['model'] == "mistralai/mistral-medium"
        
        # Check custom_llm_provider is set
        assert call_args['custom_llm_provider'] == "openrouter"
        
        # Check API configuration
        from litellm import litellm
        assert litellm.api_base == "https://openrouter.ai/api/v1/chat/completions"
        assert litellm.api_key == "test-key"
        assert litellm.drop_params is True

@pytest.mark.asyncio
async def test_openrouter_api_call():
    """Test that OpenRouter API calls are formatted correctly."""
    config = WebUIConfig(
        provider="openrouter",
        model="mistralai/mistral-medium",
        research_only=True
    )
    
    # Mock the actual API response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    
    with patch('webui.app.completion') as mock_completion:
        mock_completion.return_value = mock_response
        
        await handle_conversation("Test message", config)
        
        # Verify the API call format
        mock_completion.assert_called_once_with(
            model="mistralai/mistral-medium",
            messages=[{"role": "user", "content": "Test message"}],
            temperature=0.7,
            max_tokens=2000,
            custom_llm_provider="openrouter"
        )

def test_load_openrouter_models():
    """Test that OpenRouter models are loaded correctly."""
    # Mock API response data
    mock_api_response = {
        "data": [
            {"id": "mistralai/mistral-large"},
            {"id": "mistralai/mistral-medium"},
            {"id": "google/gemini-pro"},
            {"id": "meta/llama-2-70b-chat"}
        ]
    }
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response
    
    with patch('requests.get', return_value=mock_response), \
         patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}):
        
        # Load models
        models = load_available_models()
        
        # Verify OpenRouter models are loaded
        assert 'openrouter' in models
        assert isinstance(models['openrouter'], list)
        
        # Verify model IDs are preserved
        expected_models = [item['id'] for item in mock_api_response['data']]
        assert all(model in models['openrouter'] for model in expected_models)
        
        # Verify fallback models are included when API fails
        mock_response.status_code = 500
        models = load_available_models()
        assert 'openrouter' in models
        assert "openrouter/mistralai/mistral-large" in models['openrouter']
        assert "openrouter/mistralai/mistral-medium" in models['openrouter']
        assert "openrouter/google/gemini-pro" in models['openrouter'] 