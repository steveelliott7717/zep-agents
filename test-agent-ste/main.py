# main.py (fixed registration section)

from fastapi import FastAPI, Request
from agent_core import CoreAgent, AgentConfig, MemoryConfig, LLMConfig
from config_loader import load_agent_config
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Initialize agent instance variable
agent = None

# Get environment variables
AGENT_ID = os.environ.get("AGENT_ID", "default-agent")
APP_NAME = os.environ.get("FLY_APP_NAME", AGENT_ID)

async def register_agent_in_supabase():
    """Register agent in Supabase with proper error handling."""
    try:
        # Import here to ensure env vars are loaded
        from supabase import create_client
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logging.warning("Supabase credentials not found, skipping registration")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Check if agent already exists
        existing = supabase.table("agents_registry").select("*").eq("agent_id", AGENT_ID).execute()
        
        if existing.data:
            # Update existing record
            response = supabase.table("agents_registry").update({
                "app_name": APP_NAME,
                "status": "active",
                "last_triggered": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("agent_id", AGENT_ID).execute()
            logging.info(f"✅ Agent {AGENT_ID} updated in registry")
        else:
            # Insert new record
            response = supabase.table("agents_registry").insert({
                "agent_id": AGENT_ID,
                "app_name": APP_NAME,
                "priority": 5,
                "status": "active",
                "last_triggered": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "request_count": 0,
                "failure_count": 0,
                "total_duration_ms": 0
            }).execute()
            logging.info(f"✅ Agent {AGENT_ID} registered in registry")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to register agent in Supabase: {e}")
        logging.error(f"Error details: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    global agent
    
    # Register agent first
    await register_agent_in_supabase()
    
    # Load config from Supabase
    agent_id = AGENT_ID
    
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

    # Register Supabase plugins if credentials are available
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
        from supabase_plugin import register_supabase_plugins
        register_supabase_plugins(agent)
    
    yield
    
    # Update status on shutdown
    try:
        from supabase import create_client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if supabase_url and supabase_key:
            supabase = create_client(supabase_url, supabase_key)
            supabase.table("agents_registry").update({
                "status": "stopped",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("agent_id", AGENT_ID).execute()
    except:
        pass
    
    logging.info("Shutting down agent...")

# Rest of your FastAPI code remains the same...