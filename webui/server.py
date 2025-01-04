"""WebSocket server implementation for the WebUI."""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from webui.config import WebUIConfig
import httpx
from ra_aid.llm import initialize_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Store active connections
active_connections: List[WebSocket] = []

class TaskRequest(BaseModel):
    """Task request model."""
    content: str
    config: Dict[str, any]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/models/{provider}")
async def get_provider_models(provider: str):
    """Get available models for a specific provider."""
    try:
        config = WebUIConfig.PROVIDER_CONFIGS.get(provider)
        if not config:
            return {"error": f"Provider {provider} not configured"}
            
        if not os.getenv(config.env_key):
            return {"error": f"API key not configured for {provider}"}
            
        if config.client_library:
            # Use LangChain client
            llm = initialize_llm(provider, config.default_models[0])
            return {"models": config.default_models}
        else:
            # Use direct API call for OpenRouter
            headers = WebUIConfig.get_provider_headers(provider)
            api_url = WebUIConfig.get_provider_url(provider)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{api_url}/models", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    return {"models": [model["id"] for model in data["data"]]}
                else:
                    return {"error": f"Failed to fetch models: {response.text}"}
                    
    except Exception as e:
        logger.error(f"Error getting models for {provider}: {str(e)}")
        return {"error": str(e)}

async def handle_task(websocket: WebSocket, task: str, config: dict) -> None:
    """Handle a task request."""
    try:
        provider = config.get("provider", "anthropic")
        model = config.get("model", "claude-3-opus-20240229")
        
        # Initialize LLM based on provider
        llm = initialize_llm(provider, model)
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "content": f"Processing task using {provider}/{model}..."
        })
        
        # Process task
        response = await llm.agenerate([task])
        result = response.generations[0][0].text
        
        # Send result
        await websocket.send_json({
            "type": "success",
            "content": result
        })
        
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "content": f"Error processing task: {str(e)}"
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("Client connected")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                if message.get("type") == "task":
                    # Process task
                    await handle_task(
                        websocket,
                        message.get("content", ""),
                        message.get("config", {})
                    )
                else:
                    # Echo unknown message types
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Unknown message type: {message.get('type')}"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid JSON format"
                })
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
