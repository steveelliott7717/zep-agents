import os
from supabase import create_client
from datetime import datetime

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

AGENT_ID = os.environ.get("AGENT_ID", "my-agent")
APP_NAME = os.environ.get("FLY_APP_NAME", AGENT_ID)

def register():
    try:
        existing = supabase.table("agents_registry").select("*").eq("agent_id", AGENT_ID).execute()
        if existing.data:
            print("✅ Agent already registered.")
            return

        response = supabase.table("agents_registry").upsert({
            "agent_id": AGENT_ID,
            "app_name": APP_NAME,
            "priority": 5,
            "status": "active",
            "last_triggered": datetime.utcnow().isoformat()
        }).execute()
        print("✅ Agent registered successfully.")
    except Exception as e:
        print("❌ Failed to register agent:", e)

if __name__ == "__main__":
    register()
