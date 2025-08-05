# main.py

from fastapi import FastAPI, Request
from agent_core import CoreAgent, AgentConfig, MemoryConfig, LLMConfig
from config_loader import load_agent_config
import asyncio
from contextlib import asynccontextmanager
import logging
import os

logging.basicConfig(level=logging.INFO)

# Initialize agent instance variable
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    global agent
    
    # Load config from Supabase
    agent_id = os.environ.get("AGENT_ID", "test-agent-ste")
    
    try:
        config_data = load_agent_config(agent_id)
        logging.info(f"Loaded config for agent {agent_id}: {config_data}")
    except Exception as e:
        logging.warning(f"Failed to load config from Supabase: {e}")
        # Use default config if Supabase fails
        config_data = {
            "graphs": ["default_graph"],
            "shared_groups": ["all_agents"],
            "llm_model": "gpt-4o-mini",
            "system_prompt": "You are a helpful assistant.",
            "supabase_tables": [],
            "plugins_enabled": []
        }
    
    # Initialize agent
    agent = CoreAgent(
        AgentConfig(
            agent_id=agent_id,
            graphs=config_data.get("graphs", ["default_graph"]),
            memory=MemoryConfig(
                persistent_facts=[],  # Loaded dynamically via Zep later
                shared_memory_groups=config_data.get("shared_groups", ["all_agents"]),
                include_graph_data=True
            ),
            llm=LLMConfig(
                model=config_data.get("llm_model", "gpt-4o-mini"),
                system_instructions=config_data.get(
                    "system_prompt", "You are a helpful assistant."
                )
            ),
            metadata={
                "supabase_tables": config_data.get("supabase_tables", []),
                "plugins_enabled": config_data.get("plugins_enabled", [])
            }
        )
    )
    
    # Initialize the agent
    await agent.initialize()
    logging.info(f"✅ Agent {agent_id} initialized successfully")
    
    yield
    
    # Cleanup if needed
    logging.info("Shutting down agent...")

app = FastAPI(
    title="Agent System",
    description="Single agent deployment on Fly.io",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "agent_id": agent.agent_id if agent else "not_initialized",
        "graphs": agent.graphs if agent else []
    }

@app.post("/process")
async def process_trigger(request: Request):
    """Process a trigger for the agent"""
    if not agent:
        return {"error": "Agent not initialized"}, 503
    
    try:
        data = await request.json()
        result = await agent.process(data)
        return {"response": result}
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return {"error": str(e)}, 500

@app.get("/agent/info")
async def agent_info():
    """Get agent information"""
    if not agent:
        return {"error": "Agent not initialized"}, 503
    
    return {
        "agent_id": agent.agent_id,
        "graphs": agent.graphs,
        "memory_groups": agent.config.memory.shared_memory_groups,
        "llm_model": agent.config.llm.model
    }

@app.post("/agent/memory/fact")
async def add_fact(request: Request):
    """Add a persistent fact to agent memory"""
    if not agent:
        return {"error": "Agent not initialized"}, 503
    
    data = await request.json()
    fact = data.get("fact")
    if not fact:
        return {"error": "No fact provided"}, 400
    
    await agent.add_persistent_fact(fact)
    return {"status": "success", "message": "Fact added"}

@app.post("/agent/memory/share")
async def share_memory(request: Request):
    """Share memory with groups"""
    if not agent:
        return {"error": "Agent not initialized"}, 503
    
    data = await request.json()
    fact = data.get("fact")
    groups = data.get("groups", agent.config.memory.shared_memory_groups)
    
    if not fact:
        return {"error": "No fact provided"}, 400
    
    await agent.share_memory(fact, groups)
    return {"status": "success", "shared_with": groups}

