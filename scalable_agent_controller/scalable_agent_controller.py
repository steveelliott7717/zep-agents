# scalable_agent_controller.py
"""
Scalable agent controller that can manage thousands of agents efficiently.
Uses async patterns, connection pooling, and queuing for high throughput.
"""
import os
import asyncio
import httpx
import redis.asyncio as redis
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel
import json
from asyncio import Queue, Semaphore
from collections import defaultdict
import time
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Prefer service role for write access
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



FLY_API_TOKEN = os.getenv("FLY_API_TOKEN")
FLY_ORG = os.getenv("FLY_ORG")  # e.g., "personal"


logging.basicConfig(level=logging.INFO)

# ============== Configuration ==============

# Scaling parameters
MAX_CONCURRENT_REQUESTS = 100  # Max parallel agent requests
MAX_MACHINES_PER_APP = 10      # Max machines per app to start
REQUEST_TIMEOUT = 60           # Timeout for agent requests
HEALTH_CHECK_INTERVAL = 300    # How often to check agent health
CACHE_TTL = 60                 # Cache machine info for 60 seconds

# ============== Models ==============

class AgentRequest(BaseModel):
    agent_id: str
    priority: int = 5  # 1-10, higher = more important
    data: Dict[str, Any]
    timeout: int = 30
    callback_url: Optional[str] = None

class AgentResponse(BaseModel):
    request_id: str
    agent_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int
    machine_id: Optional[str] = None

# ============== Scalable Controller ==============

class ScalableAgentController:
    def __init__(self):
        self.api_token = os.environ.get("FLY_API_TOKEN")
        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.base_url = "https://api.machines.dev/v1"

        self.agent_registry = {}

        
        # Connection pooling for HTTP requests
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=50,
                max_connections=200,
                keepalive_expiry=30
            ),
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
        )
        
        # Concurrency control
        self.semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
        self.request_queue = Queue(maxsize=10000)
        
        # Caching and state
        self.machine_cache = {}  # app -> machines mapping
        self.cache_timestamps = {}
        self.agent_stats = defaultdict(lambda: {
            "requests": 0,
            "failures": 0,
            "total_time": 0,
            "last_used": None
        })
        
        # Redis for distributed state (optional)
        self.redis_client = None
        
    async def load_agent_registry(self):
        """Load agent registry from Supabase table 'agents_registry'."""
        try:
            response = supabase.table("agents_registry").select("*").execute()
            for row in response.data:
                agent_id = row["agent_id"]
                self.agent_registry[agent_id] = {
                    "app_name": row.get("app_name", agent_id),
                    "last_triggered": datetime.utcnow(),
                    "priority": row.get("priority", 5),
                }
            logging.info(f"✅ Loaded {len(self.agent_registry)} agents from Supabase")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load agent registry from Supabase: {e}")

    async def initialize(self):
        logging.info("[Controller] Initializing...")
        await self.load_agent_registry()
        asyncio.create_task(monitor_idle_agents(self))  # if you're using the idle monitor
        logging.info("[Controller] Initialization complete.")


    
    async def close(self):
        """Cleanup connections."""
        await self.http_client.aclose()
        if self.redis_client:
            await self.redis_client.close()
    
    # ============== Core Methods ==============
    
    async def trigger_agent(self, request: AgentRequest) -> str:
        """
        Queue an agent request for processing.
        Returns immediately with request ID.
        """
        request_id = f"{request.agent_id}_{int(time.time() * 1000)}"
        
        # Store request metadata
        await self._store_request(request_id, request)
        
        # Add to priority queue
        await self.request_queue.put((
            -request.priority,  # Negative for priority queue behavior
            request_id,
            request
        ))
        
        logging.info(f"📥 Queued request {request_id} for agent {request.agent_id}")
        return request_id
    
    async def _process_queue_worker(self, worker_id: int):
        """Worker to process queued requests."""
        while True:
            try:
                # Get highest priority request
                _, request_id, request = await self.request_queue.get()
                
                # Process with semaphore for rate limiting
                async with self.semaphore:
                    result = await self._process_agent_request(request_id, request)
                    
                    # Send callback if provided
                    if request.callback_url:
                        asyncio.create_task(
                            self._send_callback(request.callback_url, result)
                        )
                
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _process_agent_request(self, request_id: str, request: AgentRequest) -> AgentResponse:
        """Process a single agent request."""
        start_time = time.time()
        
        try:
            # Get or start machine for agent
            machine = await self._get_available_machine(request.agent_id)
            if not machine:
                raise Exception("No machines available")
            
            # Make request to agent
            agent_url = f"https://{request.agent_id}.fly.dev/process"
            
            response = await self.http_client.post(
                agent_url,
                json=request.data,
                timeout=request.timeout
            )
            response.raise_for_status()
            
            # Update stats
            duration_ms = int((time.time() - start_time) * 1000)
            await self._update_agent_stats(request.agent_id, True, duration_ms)
            
            return AgentResponse(
                request_id=request_id,
                agent_id=request.agent_id,
                status="success",
                result=response.json(),
                duration_ms=duration_ms,
                machine_id=machine["id"]
            )
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            await self._update_agent_stats(request.agent_id, False, duration_ms)
            
            return AgentResponse(
                request_id=request_id,
                agent_id=request.agent_id,
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
    
    async def _get_available_machine(self, app_name: str) -> Optional[Dict]:
        """Get an available machine, starting one if needed."""
        # Check cache first
        if await self._is_cache_valid(app_name):
            machines = self.machine_cache.get(app_name, [])
        else:
            # Fetch fresh machine list
            machines = await self._list_machines(app_name)
            self.machine_cache[app_name] = machines
            self.cache_timestamps[app_name] = time.time()
        
        # Find running machine
        for machine in machines:
            if machine.get("state") == "started":
                return machine
        
        # Start a stopped machine
        for machine in machines[:MAX_MACHINES_PER_APP]:
            if machine.get("state") == "stopped":
                await self._start_machine(app_name, machine["id"])
                # Wait briefly for startup
                await asyncio.sleep(2)
                return machine
        
        return None
    
    async def _list_machines(self, app_name: str) -> List[Dict]:
        """List all machines for an app."""
        try:
            response = await self.http_client.get(
                f"{self.base_url}/apps/{app_name}/machines"
            )
            response.raise_for_status()
            return response.json()
        except:
            return []
    
    async def _start_machine(self, app_name: str, machine_id: str):
        """Start a machine."""
        try:
            response = await self.http_client.post(
                f"{self.base_url}/apps/{app_name}/machines/{machine_id}/start"
            )
            response.raise_for_status()
            logging.info(f"✅ Started machine {machine_id} for {app_name}")
        except Exception as e:
            logging.error(f"❌ Failed to start machine: {e}")
    
    # ============== Helper Methods ==============
    
    async def _store_request(self, request_id: str, request: AgentRequest):
        """Store request metadata for tracking."""
        if self.redis_client:
            await self.redis_client.setex(
                f"request:{request_id}",
                300,  # 5 minute TTL
                json.dumps(request.dict())
            )
    
    async def _update_agent_stats(self, agent_id: str, success: bool, duration_ms: int):
        """Update agent performance statistics."""
        stats = self.agent_stats[agent_id]
        stats["requests"] += 1
        if not success:
            stats["failures"] += 1
        stats["total_time"] += duration_ms
        stats["last_used"] = datetime.now()
        
        # Store in Redis if available
        if self.redis_client:
            await self.redis_client.hset(
                f"agent:stats:{agent_id}",
                mapping={
                    "requests": stats["requests"],
                    "failures": stats["failures"],
                    "avg_time": stats["total_time"] // max(1, stats["requests"])
                }
            )
    
    async def _is_cache_valid(self, app_name: str) -> bool:
        """Check if machine cache is still valid."""
        timestamp = self.cache_timestamps.get(app_name, 0)
        return (time.time() - timestamp) < CACHE_TTL
    
    async def _send_callback(self, callback_url: str, response: AgentResponse):
        """Send async callback with results."""
        try:
            await self.http_client.post(callback_url, json=response.dict())
        except Exception as e:
            logging.error(f"Callback failed: {e}")
    
    # ============== Monitoring ==============
    
    async def _health_check_worker(self):
        """Periodic health checks on agents."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            # Check top agents
            for agent_id in list(self.agent_stats.keys())[:100]:
                try:
                    response = await self.http_client.get(
                        f"https://{agent_id}.fly.dev/status",
                        timeout=5
                    )
                    if response.status_code != 200:
                        logging.warning(f"⚠️ Agent {agent_id} health check failed")
                except:
                    pass
    
    async def _stats_reporter(self):
        """Report controller statistics."""
        while True:
            await asyncio.sleep(60)
            
            total_requests = sum(s["requests"] for s in self.agent_stats.values())
            total_failures = sum(s["failures"] for s in self.agent_stats.values())
            queue_size = self.request_queue.qsize()
            
            logging.info(f"""
            📊 Controller Stats:
            - Active Agents: {len(self.agent_stats)}
            - Total Requests: {total_requests}
            - Failed Requests: {total_failures}
            - Queue Size: {queue_size}
            - Success Rate: {((total_requests - total_failures) / max(1, total_requests)) * 100:.1f}%
            """)

# ============== FastAPI App ==============

app = FastAPI(title="Scalable Agent Controller")
controller = ScalableAgentController()

@app.on_event("startup")
async def startup():
    await controller.initialize()

@app.on_event("shutdown")
async def shutdown():
    await controller.close()

@app.get("/")
async def health():
    return {
        "status": "healthy",
        "queue_size": controller.request_queue.qsize(),
        "active_agents": len(controller.agent_stats)
    }

@app.post("/trigger")
async def trigger_agent(request: AgentRequest):
    """Queue an agent request."""
    request_id = await controller.trigger_agent(request)
    return {
        "request_id": request_id,
        "status": "queued",
        "estimated_wait": controller.request_queue.qsize() / MAX_CONCURRENT_REQUESTS
    }

@app.get("/status/{request_id}")
async def get_request_status(request_id: str):
    """Get status of a queued request."""
    if controller.redis_client:
        data = await controller.redis_client.get(f"request:{request_id}")
        if data:
            return json.loads(data)
    return {"error": "Request not found"}

@app.get("/agents/{agent_id}/stats")
async def get_agent_stats(agent_id: str):
    """Get performance stats for an agent."""
    stats = controller.agent_stats.get(agent_id)
    if not stats:
        raise HTTPException(404, "Agent not found")
    
    return {
        "agent_id": agent_id,
        "total_requests": stats["requests"],
        "failure_rate": stats["failures"] / max(1, stats["requests"]),
        "avg_response_time": stats["total_time"] / max(1, stats["requests"]),
        "last_used": stats["last_used"].isoformat() if stats["last_used"] else None
    }

@app.post("/agents/batch")
async def trigger_batch(requests: List[AgentRequest]):
    """Trigger multiple agents in batch."""
    request_ids = []
    for request in requests:
        request_id = await controller.trigger_agent(request)
        request_ids.append(request_id)
    
    return {
        "batch_size": len(requests),
        "request_ids": request_ids
    }




async def stop_fly_machine(machine_id: str):
    url = f"https://api.machines.dev/v1/apps/{machine_id}/machines"
    headers = {
        "Authorization": f"Bearer {FLY_API_TOKEN}",
        "Fly-Organization": FLY_ORG,
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            machines = resp.json()
            for machine in machines:
                mid = machine["id"]
                print(f"[Controller] Sending stop command to machine: {mid}")
                await client.post(f"{url}/{mid}/stop", headers=headers)
        except Exception as e:
            print(f"[Controller] Failed to stop agent {machine_id}: {e}")


async def monitor_idle_agents():
    while True:
        now = datetime.datetime.utcnow()
        for agent_id, metrics in controller.agent_stats.items():
            delta = now - metrics["last_triggered"]
            if delta.total_seconds() > 300:  # 5 minutes
                print(f"[Controller] Stopping idle agent: {agent_id}")
                await stop_fly_machine(agent_id)
        await asyncio.sleep(60)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)