import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from components.research import research_component
from webui.config import WebUIConfig

@pytest.fixture
def mock_initialize_llm():
    with patch('components.research.initialize_llm') as mock:
        yield mock

@pytest.fixture
def mock_run_research_agent():
    with patch('components.research.run_research_agent') as mock:
        mock.return_value = {
            'research_notes': ['note1', 'note2'],
            'key_facts': {'fact1': 'value1'},
            'success': True
        }
        yield mock

@pytest.fixture
def test_config():
    return WebUIConfig(
        provider="openai",
        model="openai/gpt-4",
        research_only=True,
        cowboy_mode=False,
        hil=True,
        web_research_enabled=True
    )

def test_research_with_webui_config(mock_initialize_llm, mock_run_research_agent, test_config):
    """Test research component with WebUIConfig object."""
    with patch('streamlit.write') as mock_write:
        result = research_component("test task", test_config)

        assert result["success"] is True
        assert len(result["research_notes"]) == 2
        assert len(result["key_facts"]) == 1
        
        # Verify LLM initialization
        mock_initialize_llm.assert_called_once_with(test_config)
        
        # Verify research agent call
        mock_run_research_agent.assert_called_once()
        call_args = mock_run_research_agent.call_args[1]
        assert call_args["expert_enabled"] is True
        assert call_args["research_only"] is True
        assert call_args["hil"] is True
        assert call_args["web_research_enabled"] is True

def test_research_error_handling(mock_initialize_llm, mock_run_research_agent, test_config):
    """Test research component error handling with WebUIConfig."""
    mock_run_research_agent.side_effect = ValueError("Test error")
    
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         pytest.raises(ValueError, match="Test error"):
        
        research_component("test task", test_config)
        mock_error.assert_called()

def test_research_null_results(mock_initialize_llm, mock_run_research_agent, test_config):
    """Test research component handling of null results."""
    mock_run_research_agent.return_value = None
    
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         pytest.raises(ValueError, match="Research agent returned no results"):
        
        research_component("test task", test_config)

def test_research_result_processing(mock_initialize_llm, mock_run_research_agent, test_config):
    """Test research component result processing."""
    mock_run_research_agent.return_value = """
Research Notes:
- Note 1
- Note 2

Key Facts:
- fact1: value1
- fact2: value2
"""
    
    with patch('streamlit.write') as mock_write:
        result = research_component("test task", test_config)
        
        assert len(result["research_notes"]) == 2
        assert len(result["key_facts"]) == 2
        assert "Note 1" in result["research_notes"]
        assert result["key_facts"]["fact1"] == "value1" 