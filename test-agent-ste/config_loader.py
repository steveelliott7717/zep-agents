# config_loader.py

import os
import logging

logging.basicConfig(level=logging.INFO)

def load_agent_config(agent_id: str) -> dict:
    """Load structured agent config from Supabase."""
    try:
        from supabase import create_client
        
        # Get Supabase credentials
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            logging.warning("Supabase credentials not found, using default config")
            return get_default_config(agent_id)
        
        # Initialize Supabase client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Fetch config
        response = client.table("agent_config").select("*").eq("agent_id", agent_id).execute()
        rows = response.data or []
        
        if not rows:
            logging.info(f"No config found for agent {agent_id}, using defaults")
            return get_default_config(agent_id)
        
        config = {row["config_key"]: row["config_value"] for row in rows}
        return config
        
    except ImportError:
        logging.warning("Supabase not installed, using default config")
        return get_default_config(agent_id)
    except Exception as e:
        logging.error(f"Error loading config for agent {agent_id}: {e}")
        return get_default_config(agent_id)

def get_default_config(agent_id: str) -> dict:
    """Return default configuration for an agent."""
    return {
        "graphs": ["knowledge_base", "memory"],
        "shared_groups": ["all_agents"],
        "llm_model": "gpt-4o-mini",
        "system_prompt": f"You are agent {agent_id}. You are a helpful AI assistant.",
        "supabase_tables": [],
        "plugins_enabled": ["chat"]
    }