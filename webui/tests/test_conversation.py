import pytest
from unittest.mock import patch, MagicMock
from webui.app import handle_conversation
from webui.config import WebUIConfig

@pytest.mark.asyncio
async def test_openrouter_claude_model_handling():
    """Test that OpenRouter Claude models are handled correctly without trying to use Anthropic API."""
    config = WebUIConfig(
        provider="openrouter",
        model="anthropic/claude-3.5-haiku-20241022:beta",
        research_only=True
    )
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    
    with patch('webui.app.completion') as mock_completion:
        mock_completion.return_value = mock_response
        
        await handle_conversation("Test message", config)
        
        # Verify completion was called with correct model name
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        
        assert call_args['model'] == "anthropic/claude-3.5-haiku-20241022:beta"
        assert not call_args['model'].startswith("openrouter/")
        assert call_args['messages'] == [{"role": "user", "content": "Test message"}]

@pytest.mark.asyncio
async def test_openrouter_non_claude_model_handling():
    """Test that other OpenRouter models are handled correctly."""
    config = WebUIConfig(
        provider="openrouter",
        model="openrouter/mistralai/mistral-medium",
        research_only=True
    )
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    
    with patch('webui.app.completion') as mock_completion:
        mock_completion.return_value = mock_response
        
        await handle_conversation("Test message", config)
        
        # Verify completion was called with correct model name
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        
        assert call_args['model'] == "openrouter/mistralai/mistral-medium"
        assert call_args['messages'] == [{"role": "user", "content": "Test message"}]

@pytest.mark.asyncio
async def test_anthropic_direct_model_handling():
    """Test that Anthropic models are handled correctly when using Anthropic directly."""
    config = WebUIConfig(
        provider="anthropic",
        model="anthropic/claude-3-opus@20240229",
        research_only=True
    )
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    
    with patch('webui.app.completion') as mock_completion:
        mock_completion.return_value = mock_response
        
        await handle_conversation("Test message", config)
        
        # Verify completion was called with correct model name
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args[1]
        
        assert call_args['model'] == "anthropic/claude-3-opus@20240229"
        assert call_args['messages'] == [{"role": "user", "content": "Test message"}] 