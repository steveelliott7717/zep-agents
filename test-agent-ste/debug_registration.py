# debug_registration.py
"""
Script to debug agent registration issues
"""
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

async def debug_registration():
    """Debug the registration process."""
    
    # Check environment variables
    print("=== Environment Variables ===")
    print(f"SUPABASE_URL: {'✅ Set' if os.environ.get('SUPABASE_URL') else '❌ Missing'}")
    print(f"SUPABASE_KEY: {'✅ Set' if os.environ.get('SUPABASE_KEY') else '❌ Missing'}")
    print(f"AGENT_ID: {os.environ.get('AGENT_ID', 'default-agent')}")
    print(f"FLY_APP_NAME: {os.environ.get('FLY_APP_NAME', 'Not set')}")
    
    if not os.environ.get('SUPABASE_URL') or not os.environ.get('SUPABASE_KEY'):
        print("\n❌ Missing Supabase credentials. Set them with:")
        print("fly secrets set SUPABASE_URL='your-url' SUPABASE_KEY='your-key'")
        return
    
    try:
        from supabase import create_client
        
        supabase = create_client(
            os.environ.get('SUPABASE_URL'),
            os.environ.get('SUPABASE_KEY')
        )
        
        print("\n=== Testing Supabase Connection ===")
        
        # Test connection by listing tables
        # This is a simple query to verify connection
        test_query = supabase.table("agents_registry").select("*").limit(1).execute()
        print("✅ Connected to Supabase successfully")
        
        # Try to register
        agent_id = os.environ.get("AGENT_ID", "test-agent")
        app_name = os.environ.get("FLY_APP_NAME", agent_id)
        
        print(f"\n=== Attempting Registration ===")
        print(f"Agent ID: {agent_id}")
        print(f"App Name: {app_name}")
        
        # Check if exists
        existing = supabase.table("agents_registry").select("*").eq("agent_id", agent_id).execute()
        
        if existing.data:
            print("ℹ️ Agent already exists, updating...")
            response = supabase.table("agents_registry").update({
                "app_name": app_name,
                "status": "active",
                "last_triggered": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).eq("agent_id", agent_id).execute()
        else:
            print("ℹ️ Creating new agent registration...")
            response = supabase.table("agents_registry").insert({
                "agent_id": agent_id,
                "app_name": app_name,
                "priority": 5,
                "status": "active",
                "last_triggered": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "request_count": 0,
                "failure_count": 0,
                "total_duration_ms": 0
            }).execute()
        
        print("✅ Registration successful!")
        print(f"Response: {response.data}")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        
        # Common issues
        if "relation" in str(e) and "does not exist" in str(e):
            print("\n💡 The table 'agents_registry' doesn't exist. Create it with the SQL schema provided.")
        elif "permission" in str(e):
            print("\n💡 Permission issue. Make sure you're using the service role key (not anon key).")
        elif "column" in str(e):
            print("\n💡 Column mismatch. Check if your table schema matches the expected columns.")

if __name__ == "__main__":
    asyncio.run(debug_registration())