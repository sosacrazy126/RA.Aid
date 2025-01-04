import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from components.planning import planning_component
from webui.config import WebUIConfig

@pytest.fixture
def mock_initialize_llm():
    with patch('components.planning.initialize_llm') as mock:
        yield mock

@pytest.fixture
def mock_run_planning_agent():
    with patch('components.planning.run_planning_agent') as mock:
        mock.return_value = {
            'plan': ['step1', 'step2'],
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

def test_planning_with_webui_config(mock_initialize_llm, mock_run_planning_agent, test_config):
    """Test planning component with WebUIConfig object."""
    with patch('streamlit.write') as mock_write:
        result = planning_component("test task", test_config)

        assert result["success"] is True
        assert len(result["plan"]) == 2
        
        # Verify LLM initialization
        mock_initialize_llm.assert_called_once_with(test_config)
        
        # Verify planning agent call
        mock_run_planning_agent.assert_called_once()
        call_args = mock_run_planning_agent.call_args[1]
        assert call_args["hil"] is True
        assert not call_args["cowboy_mode"]

def test_planning_error_handling(mock_initialize_llm, mock_run_planning_agent, test_config):
    """Test planning component error handling with WebUIConfig."""
    mock_run_planning_agent.side_effect = ValueError("Test error")
    
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         pytest.raises(ValueError, match="Test error"):
        
        planning_component("test task", test_config)
        mock_error.assert_called()

def test_planning_null_results(mock_initialize_llm, mock_run_planning_agent, test_config):
    """Test planning component handling of null results."""
    mock_run_planning_agent.return_value = None
    
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         pytest.raises(ValueError, match="Planning agent returned no results"):
        
        planning_component("test task", test_config)

def test_planning_result_processing(mock_initialize_llm, mock_run_planning_agent, test_config):
    """Test planning component result processing."""
    mock_run_planning_agent.return_value = """
Plan:
1. Initialize project structure
2. Set up dependencies
3. Implement core functionality
4. Add tests
5. Document API
"""
    
    with patch('streamlit.write') as mock_write:
        result = planning_component("test task", test_config)
        
        assert len(result["plan"]) == 5
        assert "Initialize project structure" in result["plan"]
        assert result["success"] is True

def test_planning_with_cowboy_mode(mock_initialize_llm, mock_run_planning_agent):
    """Test planning component with cowboy mode enabled."""
    cowboy_config = WebUIConfig(
        provider="openai",
        model="openai/gpt-4",
        research_only=False,
        cowboy_mode=True,
        hil=False,
        web_research_enabled=True
    )
    
    with patch('streamlit.write') as mock_write:
        result = planning_component("test task", cowboy_config)
        
        # Verify planning agent call with cowboy mode
        call_args = mock_run_planning_agent.call_args[1]
        assert call_args["cowboy_mode"] is True
        assert not call_args["hil"] 