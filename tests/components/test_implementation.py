import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from components.implementation import implementation_component
from webui.config import WebUIConfig

@pytest.fixture
def mock_initialize_llm():
    with patch('components.implementation.initialize_llm') as mock:
        yield mock

@pytest.fixture
def mock_run_implementation_agent():
    with patch('components.implementation.run_implementation_agent') as mock:
        mock.return_value = {
            'code_changes': [
                {'file': 'test.py', 'content': 'def test(): pass'},
            ],
            'success': True
        }
        yield mock

@pytest.fixture
def test_config():
    return WebUIConfig(
        provider="openai",
        model="openai/gpt-4",
        research_only=False,
        cowboy_mode=False,
        hil=True,
        web_research_enabled=True
    )

def test_implementation_with_webui_config(mock_initialize_llm, mock_run_implementation_agent, test_config):
    """Test implementation component with WebUIConfig object."""
    with patch('streamlit.write') as mock_write:
        result = implementation_component("test task", test_config)

        assert result["success"] is True
        assert len(result["code_changes"]) == 1
        
        # Verify LLM initialization
        mock_initialize_llm.assert_called_once_with(test_config)
        
        # Verify implementation agent call
        mock_run_implementation_agent.assert_called_once()
        call_args = mock_run_implementation_agent.call_args[1]
        assert call_args["hil"] is True
        assert not call_args["cowboy_mode"]

def test_implementation_error_handling(mock_initialize_llm, mock_run_implementation_agent, test_config):
    """Test implementation component error handling with WebUIConfig."""
    mock_run_implementation_agent.side_effect = ValueError("Test error")
    
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         pytest.raises(ValueError, match="Test error"):
        
        implementation_component("test task", test_config)
        mock_error.assert_called()

def test_implementation_null_results(mock_initialize_llm, mock_run_implementation_agent, test_config):
    """Test implementation component handling of null results."""
    mock_run_implementation_agent.return_value = None
    
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         pytest.raises(ValueError, match="Implementation agent returned no results"):
        
        implementation_component("test task", test_config)

def test_implementation_result_processing(mock_initialize_llm, mock_run_implementation_agent, test_config):
    """Test implementation component result processing."""
    mock_run_implementation_agent.return_value = {
        'code_changes': [
            {'file': 'main.py', 'content': 'def main(): return True'},
            {'file': 'test.py', 'content': 'def test_main(): assert main()'},
        ],
        'success': True
    }
    
    with patch('streamlit.write') as mock_write:
        result = implementation_component("test task", test_config)
        
        assert len(result["code_changes"]) == 2
        assert any(change["file"] == "main.py" for change in result["code_changes"])
        assert result["success"] is True

def test_implementation_with_cowboy_mode(mock_initialize_llm, mock_run_implementation_agent):
    """Test implementation component with cowboy mode enabled."""
    cowboy_config = WebUIConfig(
        provider="openai",
        model="openai/gpt-4",
        research_only=False,
        cowboy_mode=True,
        hil=False,
        web_research_enabled=True
    )
    
    with patch('streamlit.write') as mock_write:
        result = implementation_component("test task", cowboy_config)
        
        # Verify implementation agent call with cowboy mode
        call_args = mock_run_implementation_agent.call_args[1]
        assert call_args["cowboy_mode"] is True
        assert not call_args["hil"]

def test_implementation_file_validation(mock_initialize_llm, mock_run_implementation_agent, test_config):
    """Test implementation component file validation."""
    mock_run_implementation_agent.return_value = {
        'code_changes': [
            {'file': '../outside.py', 'content': 'malicious_code()'},
            {'file': 'valid.py', 'content': 'valid_code()'},
        ],
        'success': True
    }
    
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error:
        result = implementation_component("test task", test_config)
        
        # Should only include valid file changes
        assert len(result["code_changes"]) == 1
        assert result["code_changes"][0]["file"] == "valid.py"
        mock_error.assert_called() 