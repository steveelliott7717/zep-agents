# supabase_plugin.py
"""
Supabase plugin for agent to read/write to tables.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from supabase import create_client

logging.basicConfig(level=logging.INFO)

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def get_supabase_client():
    """Get Supabase client with error handling."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.warning("Supabase credentials not found")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ============== Supabase Plugin Functions ==============

async def supabase_write_plugin(agent, trigger: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plugin to write data to Supabase tables.
    
    Trigger format:
    {
        "type": "supabase_write",
        "data": {
            "table": "table_name",
            "record": {"column": "value", ...},
            "operation": "insert"  # or "upsert"
        }
    }
    """
    client = get_supabase_client()
    if not client:
        return {"error": "Supabase not configured"}
    
    data = trigger.get("data", {})
    table_name = data.get("table")
    record = data.get("record", {})
    operation = data.get("operation", "insert")
    
    if not table_name or not record:
        return {"error": "Missing table name or record data"}
    
    try:
        # Add metadata
        record["agent_id"] = agent.agent_id
        record["created_at"] = datetime.now().isoformat()
        
        if operation == "upsert":
            response = client.table(table_name).upsert(record).execute()
        else:
            response = client.table(table_name).insert(record).execute()
        
        logging.info(f"✅ Agent {agent.agent_id} wrote to {table_name}")
        
        # Also store in agent's graph for tracking
        await agent.write_to_graphs({
            "action": "supabase_write",
            "table": table_name,
            "record_id": response.data[0].get("id") if response.data else None,
            "timestamp": datetime.now().isoformat()
        }, "json")
        
        return {
            "status": "success",
            "table": table_name,
            "data": response.data
        }
        
    except Exception as e:
        logging.error(f"❌ Supabase write error: {e}")
        return {"error": str(e)}

async def supabase_read_plugin(agent, trigger: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plugin to read data from Supabase tables.
    
    Trigger format:
    {
        "type": "supabase_read",
        "data": {
            "table": "table_name",
            "filters": {"column": "value"},
            "limit": 10
        }
    }
    """
    client = get_supabase_client()
    if not client:
        return {"error": "Supabase not configured"}
    
    data = trigger.get("data", {})
    table_name = data.get("table")
    filters = data.get("filters", {})
    limit = data.get("limit", 10)
    
    if not table_name:
        return {"error": "Missing table name"}
    
    try:
        query = client.table(table_name).select("*")
        
        # Apply filters
        for column, value in filters.items():
            query = query.eq(column, value)
        
        # Apply limit
        query = query.limit(limit)
        
        response = query.execute()
        
        return {
            "status": "success",
            "table": table_name,
            "count": len(response.data),
            "data": response.data
        }
        
    except Exception as e:
        logging.error(f"❌ Supabase read error: {e}")
        return {"error": str(e)}

async def supabase_process_plugin(agent, trigger: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plugin to process data and write insights to Supabase.
    
    This combines LLM processing with Supabase storage.
    """
    data = trigger.get("data", {})
    source_table = data.get("source_table")
    target_table = data.get("target_table", "processed_insights")
    process_type = data.get("process_type", "analysis")
    
    # Read data from source table
    read_result = await supabase_read_plugin(agent, {
        "type": "supabase_read",
        "data": {
            "table": source_table,
            "filters": data.get("filters", {}),
            "limit": data.get("limit", 50)
        }
    })
    
    if read_result.get("error"):
        return read_result
    
    source_data = read_result.get("data", [])
    
    # Process with LLM
    prompt = f"""
    Analyze this data from {source_table} for {process_type}:
    
    Data ({len(source_data)} records):
    {json.dumps(source_data[:10], indent=2)}  # Sample
    
    Provide insights as JSON with structure:
    {{
        "summary": "overall summary",
        "key_findings": ["finding1", "finding2"],
        "recommendations": ["rec1", "rec2"],
        "metrics": {{"metric": value}}
    }}
    """
    
    llm_response = await agent.process_with_llm(prompt, save_to_thread=False)
    
    try:
        insights = json.loads(llm_response)
    except:
        insights = {"raw_analysis": llm_response}
    
    # Write insights to target table
    write_result = await supabase_write_plugin(agent, {
        "type": "supabase_write",
        "data": {
            "table": target_table,
            "record": {
                "source_table": source_table,
                "process_type": process_type,
                "record_count": len(source_data),
                "insights": insights,
                "processed_by": agent.agent_id
            }
        }
    })
    
    return {
        "status": "processed",
        "source": source_table,
        "target": target_table,
        "records_processed": len(source_data),
        "write_result": write_result
    }

# ============== Webhook Handler ==============

async def supabase_webhook_plugin(agent, trigger: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle Supabase webhooks for real-time data processing.
    
    Webhook payload format:
    {
        "type": "INSERT" | "UPDATE" | "DELETE",
        "table": "table_name",
        "record": {...},
        "old_record": {...}  # for UPDATE
    }
    """
    webhook_data = trigger.get("data", {})
    event_type = webhook_data.get("type")
    table = webhook_data.get("table")
    record = webhook_data.get("record", {})
    
    # Process based on event type
    if event_type == "INSERT":
        # New record - analyze and enrich
        prompt = f"""
        New record in {table}:
        {json.dumps(record, indent=2)}
        
        Analyze and suggest:
        1. Data quality score (1-10)
        2. Enrichment opportunities
        3. Related actions needed
        """
        
        analysis = await agent.process_with_llm(prompt, save_to_thread=False)
        
        # Write analysis back
        await supabase_write_plugin(agent, {
            "type": "supabase_write",
            "data": {
                "table": f"{table}_analysis",
                "record": {
                    "original_id": record.get("id"),
                    "analysis": analysis,
                    "event_type": event_type
                }
            }
        })
        
    elif event_type == "UPDATE":
        old_record = webhook_data.get("old_record", {})
        # Track changes
        changes = {k: {"old": old_record.get(k), "new": v} 
                  for k, v in record.items() 
                  if old_record.get(k) != v}
        
        await agent.share_memory(
            f"Record updated in {table}: {json.dumps(changes)}",
            groups=["data_team"]
        )
    
    return {
        "status": "processed",
        "event": event_type,
        "table": table
    }

# ============== Helper Functions ==============

def register_supabase_plugins(agent):
    """Register all Supabase plugins with an agent."""
    agent.register_plugin("supabase_write", supabase_write_plugin)
    agent.register_plugin("supabase_read", supabase_read_plugin)
    agent.register_plugin("supabase_process", supabase_process_plugin)
    agent.register_plugin("supabase_webhook", supabase_webhook_plugin)
    logging.info(f"✅ Registered Supabase plugins for agent {agent.agent_id}")