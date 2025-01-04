import pytest
from unittest.mock import Mock, AsyncMock
from langchain_core.messages import AIMessage
from ra_aid.agent_utils import run_agent_with_retry

@pytest.mark.asyncio
async def test_agent_output_format():
    """Test that agent output is returned as a plain string."""
    # Mock agent
    mock_agent = Mock()
    mock_agent.invoke = AsyncMock(return_value=AIMessage(content="Test response from agent"))
    
    # Mock config
    config = {
        "configurable": {"thread_id": "test-thread"},
        "recursion_limit": 100
    }
    
    # Run agent
    result = await run_agent_with_retry(mock_agent, "Test prompt", config)
    
    # Verify output
    assert isinstance(result, str)
    assert result == "Test response from agent"
    
    # Verify agent was called correctly
    mock_agent.invoke.assert_called_once()
    call_args = mock_agent.invoke.call_args[0][0]
    assert call_args["messages"][0].content == "Test prompt"

@pytest.mark.asyncio
async def test_agent_output_without_content():
    """Test agent output when response has no content attribute."""
    # Mock agent
    mock_agent = Mock()
    mock_agent.invoke = AsyncMock(return_value={"some": "response"})
    
    # Mock config
    config = {
        "configurable": {"thread_id": "test-thread"},
        "recursion_limit": 100
    }
    
    # Run agent
    result = await run_agent_with_retry(mock_agent, "Test prompt", config)
    
    # Verify default success message is returned
    assert result == "Agent run completed successfully" 