# agent_core.py
"""
Core agent framework with flexible memory, graph management, and context assembly.
"""
import os
import asyncio
import uuid
import json
from typing import Optional, Dict, Any, List, Set, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from zep_cloud.client import AsyncZep
from zep_cloud.types import Message
import logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Initialize shared clients
client = AsyncZep(api_key=os.environ.get('ZEP_API_KEY'))
openai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# ============== Configuration Classes ==============

@dataclass
class MemoryConfig:
    """Configuration for agent memory management."""
    persistent_facts: List[str] = field(default_factory=list)  # Facts that persist across all sessions
    shared_memory_groups: List[str] = field(default_factory=list)  # Memory groups this agent belongs to
    context_window_size: int = 10  # Number of recent messages to include
    include_graph_data: bool = True  # Whether to search graphs for context

@dataclass
class LLMConfig:
    """Configuration for LLM interaction."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 500
    system_instructions: str = "You are a helpful AI assistant."

@dataclass
class AgentConfig:
    """Complete configuration for an agent."""
    agent_id: str
    graphs: List[str] = field(default_factory=list)  # Graphs this agent can access
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============== Core Agent Class ==============

class CoreAgent:
    """Base agent class with memory, graph, and context management."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.graphs = config.graphs
        self.thread_id = None
        self._plugins: Dict[str, Callable] = {}  # For modular functionality
        self._shared_memory_cache: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize the agent and create necessary resources."""
        # Create agent as a Zep user (minimal info needed)
        try:
            await client.user.add(
                user_id=self.agent_id,
                email=f"{self.agent_id}@agent.system",
                first_name=self.agent_id,
                last_name="Agent"
            )
            logging.info(f"✅ Initialized agent: {self.agent_id}")
        except:
            # Agent already exists, that's fine
            pass
            
        # Create default thread if none exists
        if not self.thread_id:
            self.thread_id = await self._create_thread()
            
        return True
    
    async def _create_thread(self, thread_name: Optional[str] = None) -> str:
        """Create a new thread for this agent."""
        thread_id = f"{self.agent_id}_{thread_name or 'default'}_{uuid.uuid4().hex[:8]}"
        try:
            await client.thread.create(
                thread_id=thread_id,
                user_id=self.agent_id,
            )
            return thread_id
        except Exception as e:
            logging.info(f"❌ Error creating thread: {e}")
            return None
    
    # ============== Memory Management ==============
    
    async def add_persistent_fact(self, fact: str):
        """Add a fact to agent's persistent memory."""
        self.config.memory.persistent_facts.append(fact)
        # Also store in graph for persistence
        await self._write_to_graph(
            f"{self.agent_id}_persistent_facts",
            {"fact": fact, "timestamp": datetime.now().isoformat()},
            "json"
        )
    
    async def get_shared_memory(self, group: str) -> List[Dict[str, Any]]:
        """Get shared memory from a specific group."""
        if group not in self.config.memory.shared_memory_groups:
            return []
            
        # Check cache first
        if group in self._shared_memory_cache:
            return self._shared_memory_cache[group]
            
        # Search shared memory graph
        results = await self._search_graph(
            f"shared_memory_{group}",
            "*",
            scope="nodes",
            limit=50
        )
        
        self._shared_memory_cache[group] = results
        return results
    
    async def share_memory(self, fact: str, groups: Optional[List[str]] = None):
        """Share a fact with memory groups."""
        share_groups = groups or self.config.memory.shared_memory_groups
        
        for group in share_groups:
            await self._write_to_graph(
                f"shared_memory_{group}",
                {
                    "fact": fact,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                },
                "json"
            )
    
    # ============== Context Assembly ==============
    
    async def assemble_context(self, 
                             trigger_data: Optional[Dict[str, Any]] = None,
                             include_recent_messages: bool = True,
                             custom_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Assemble complete context for LLM interaction.
        
        Returns a structured context with all relevant information.
        """
        context = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "system_instructions": self.config.llm.system_instructions,
            "persistent_facts": self.config.memory.persistent_facts,
            "shared_memory": {},
            "recent_messages": [],
            "graph_data": {},
            "trigger_data": trigger_data or {},
            "custom_context": custom_context
        }
        
        # Get shared memory from all groups
        for group in self.config.memory.shared_memory_groups:
            context["shared_memory"][group] = await self.get_shared_memory(group)
        
        # Get recent conversation history if requested
        if include_recent_messages and self.thread_id:
            try:
                memory = await client.thread.get_user_context(thread_id=self.thread_id)
                if memory and memory.context:
                    context["recent_messages"] = memory.context
            except:
                pass
        
        # Search graphs for relevant data
        if self.config.memory.include_graph_data and trigger_data:
            query = trigger_data.get("query", "")
            if query:
                for graph_id in self.graphs:
                    results = await self._search_graph(graph_id, query)
                    if results:
                        context["graph_data"][graph_id] = results
        
        return context
    
    async def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format the assembled context into a string for LLM."""
        formatted = []
        
        # System instructions
        formatted.append(f"System Instructions:\n{context['system_instructions']}\n")
        
        # Persistent facts
        if context['persistent_facts']:
            formatted.append("Persistent Knowledge:")
            for fact in context['persistent_facts']:
                formatted.append(f"- {fact}")
            formatted.append("")
        
        # Shared memory
        if any(context['shared_memory'].values()):
            formatted.append("Shared Context:")
            for group, memories in context['shared_memory'].items():
                if memories:
                    formatted.append(f"\n[{group}]")
                    for memory in memories:
                        formatted.append(f"- {memory}")
            formatted.append("")
        
        # Recent messages
        if context['recent_messages']:
            formatted.append(f"Recent Context:\n{context['recent_messages']}\n")
        
        # Graph data
        if context['graph_data']:
            formatted.append("Relevant Data:")
            for graph_id, data in context['graph_data'].items():
                formatted.append(f"\n[Graph: {graph_id}]")
                for item in data[:5]:  # Limit to avoid context overflow
                    formatted.append(f"- {item}")
            formatted.append("")
        
        # Custom context
        if context['custom_context']:
            formatted.append(f"Additional Context:\n{context['custom_context']}\n")
        
        return "\n".join(formatted)
    
    # ============== Graph Operations ==============
    
    async def _write_to_graph(self, graph_id: str, data: Any, data_type: str = "text"):
        """Write data to a specific graph."""
        try:
            if data_type == "json" and isinstance(data, dict):
                data = json.dumps(data)
                
            await client.graph.add(
                graph_id=graph_id,
                type=data_type,
                data=data
            )
            return True
        except Exception as e:
            logging.info(f"❌ Error writing to graph {graph_id}: {e}")
            return False
    
    async def _search_graph(self, graph_id: str, query: str, scope: str = "edges", limit: int = 10):
        """Search a specific graph."""
        try:
            results = await client.graph.search(
                graph_id=graph_id,
                query=query,
                scope=scope,
                limit=limit
            )
            return results
        except:
            return []
    
    async def read_from_graphs(self, query: str, graph_ids: Optional[List[str]] = None):
        """Read data from multiple graphs."""
        target_graphs = graph_ids or self.graphs
        all_results = {}
        
        for graph_id in target_graphs:
            results = await self._search_graph(graph_id, query)
            if results:
                all_results[graph_id] = results
                
        return all_results
    
    async def write_to_graphs(self, data: Any, data_type: str = "text", graph_ids: Optional[List[str]] = None):
        """Write data to multiple graphs."""
        target_graphs = graph_ids or self.graphs
        results = {}
        
        for graph_id in target_graphs:
            success = await self._write_to_graph(graph_id, data, data_type)
            results[graph_id] = success
            
        return results
    
    # ============== LLM Interaction ==============
    
    async def process_with_llm(self, 
                              user_input: str,
                              context: Optional[Dict[str, Any]] = None,
                              save_to_thread: bool = True) -> str:
        """
        Process input with LLM using assembled context.
        """
        # Assemble context if not provided
        if not context:
            context = await self.assemble_context(
                trigger_data={"query": user_input},
                include_recent_messages=save_to_thread
            )
        
        # Format context for LLM
        context_str = await self.format_context_for_llm(context)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": context_str},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = await openai_client.chat.completions.create(
                model=self.config.llm.model,
                messages=messages,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
            
            assistant_response = response.choices[0].message.content
            
            # Save to thread if requested
            if save_to_thread and self.thread_id:
                await client.thread.add_messages(
                    self.thread_id,
                    messages=[
                        Message(
                            name="User",
                            content=user_input,
                            role="user"
                        ),
                        Message(
                            name=self.agent_id,
                            content=assistant_response,
                            role="assistant"
                        )
                    ]
                )
            
            return assistant_response
            
        except Exception as e:
            logging.info(f"❌ LLM Error: {e}")
            return f"Error processing request: {str(e)}"
    
    # ============== Plugin System ==============
    
    def register_plugin(self, name: str, func: Callable):
        """Register a plugin function for modular functionality."""
        self._plugins[name] = func
    
    async def execute_plugin(self, name: str, *args, **kwargs):
        """Execute a registered plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not registered")
        
        plugin_func = self._plugins[name]
        if asyncio.iscoroutinefunction(plugin_func):
            return await plugin_func(self, *args, **kwargs)
        else:
            return plugin_func(self, *args, **kwargs)
    
    # ============== Main Execution ==============
    
    async def process(self, trigger: Dict[str, Any]) -> Any:
        """
        Main processing method - override in subclasses or use plugins.
        
        Args:
            trigger: Dictionary containing trigger type and data
                    e.g., {"type": "chat", "data": {"message": "Hello"}}
                         {"type": "schedule", "data": {"task": "daily_summary"}}
        """
        trigger_type = trigger.get("type", "unknown")
        
        # Check if we have a plugin for this trigger type
        if trigger_type in self._plugins:
            return await self.execute_plugin(trigger_type, trigger)
        
        # Default behavior - process with LLM
        user_input = trigger.get("data", {}).get("message", "")
        if user_input:
            return await self.process_with_llm(user_input)
        
        return {"status": "no_handler", "trigger": trigger}


# ============== Helper Functions ==============

async def create_shared_memory_graph(group_name: str, description: str):
    """Create a shared memory graph for agent groups."""
    graph_id = f"shared_memory_{group_name}"
    try:
        await client.graph.create(
            graph_id=graph_id,
            name=f"Shared Memory: {group_name}",
            description=description
        )
        logging.info(f"✅ Created shared memory graph: {graph_id}")
    except:
        logging.info(f"ℹ️ Shared memory graph {graph_id} already exists")


# ============== Example Usage ==============

async def example_usage():
    """Example of how to use the core agent."""
    
    # Define agent configuration
    config = AgentConfig(
        agent_id="agent_001",
        graphs=["knowledge_base", "task_queue", "analytics"],
        memory=MemoryConfig(
            persistent_facts=[
                "I am a helpful assistant",
                "I have access to company knowledge base"
            ],
            shared_memory_groups=["customer_service", "all_agents"],
            context_window_size=10,
            include_graph_data=True
        ),
        llm=LLMConfig(
            model="gpt-4o-mini",
            temperature=0.7,
            system_instructions="""You are a customer service agent.
            Be helpful, professional, and empathetic.
            Use the knowledge base to answer questions accurately."""
        )
    )
    
    # Create agent
    agent = CoreAgent(config)
    await agent.initialize()
    
    # Add a persistent fact
    await agent.add_persistent_fact("Our business hours are 9 AM to 5 PM EST")
    
    # Share memory with other agents
    await agent.share_memory(
        "New product launch scheduled for next month",
        groups=["customer_service", "sales"]
    )
    
    # Process a user query
    response = await agent.process({
        "type": "chat",
        "data": {"message": "What are your business hours?"}
    })
    
    logging.info(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(example_usage())