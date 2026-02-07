"""
The Briefing - Shared Tools

Common tool definitions and executors used across multiple agents.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

from google.genai import types

from utils import logger


# =============================================================================
# DATABASE INFRASTRUCTURE
# =============================================================================

def get_supabase_client():
    """
    Create Supabase client from configuration.
    
    Shared by aggregation and agent tools.
    """
    from supabase import create_client
    from config import get_config
    
    config = get_config()
    if not config.supabase_url or not config.supabase_service_key:
        raise ValueError("Supabase configuration (SUPABASE_URL, SUPABASE_SERVICE_KEY) missing")
    
    return create_client(config.supabase_url, config.supabase_service_key)


# =============================================================================
# SEARCH RESULT STORAGE
# =============================================================================

@dataclass
class SearchResult:
    """
    Complete search result with full content for verification.
    
    Stores everything returned from Tavily for transparency and CoVe verification.
    """
    query: str
    title: str
    url: str
    content: str  # Full text content from the page
    publisher: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    score: float = 0.0  # Tavily relevance score
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "publisher": self.publisher,
            "timestamp": self.timestamp,
            "score": self.score,
        }


# Global storage for search results during a pipeline run
# Reset at the start of each run
_search_results: list[SearchResult] = []


def reset_search_results() -> None:
    """Reset search results storage. Call at start of each pipeline run."""
    global _search_results
    _search_results = []
    logger.info("   🔍 Search results storage reset")


def get_all_search_results() -> list[SearchResult]:
    """Get all search results from current pipeline run."""
    return _search_results.copy()


def get_search_results_for_verification() -> str:
    """
    Get all search content as a single string for CoVe verification.
    
    Returns concatenated titles and content from all searches.
    """
    parts = []
    for result in _search_results:
        parts.append(f"{result.title} {result.content}")
    return " ".join(parts)


# =============================================================================
# GRAPH TOOLS
# =============================================================================

def get_graph_tools() -> list[types.FunctionDeclaration]:
    """
    Get the set of graph-querying tools for the Constellation knowledge graph.
    
    Returns a list of FunctionDeclarations for the Analyst agent.
    """
    return [
        types.FunctionDeclaration(
            name="get_event_graph",
            description="""Get the relationship network around THIS specific event.
Shows all entities involved and their relationships (supply chains, dependencies, alliances).
PRIORITY TOOL: Call this FIRST to understand the network topology.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "event_id": types.Schema(
                        type=types.Type.STRING,
                        description="The ID of the event to analyze."
                    ),
                    "include_indirect": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="Whether to include indirect relationships (1 hop away). Default: false"
                    ),
                },
                required=["event_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_entity_relationships",
            description="""Get all known relationships for an entity from our knowledge graph.
Use to understand existing connections between countries, organizations, or companies.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "entity_name": types.Schema(
                        type=types.Type.STRING,
                        description="The entity name (e.g., 'Russia', 'TSMC', 'NATO')."
                    ),
                    "limit": types.Schema(
                        type=types.Type.NUMBER,
                        description="Max number of relationships (default: 10, max: 20)."
                    ),
                },
                required=["entity_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_causal_chain",
            description="""Trace the chain of events and factors that LED TO this event.
Use to answer 'Why did this happen?' or 'What's the background?'.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "event_id": types.Schema(
                        type=types.Type.STRING,
                        description="The event ID to trace back from."
                    ),
                    "max_depth": types.Schema(
                        type=types.Type.NUMBER,
                        description="Steps back to trace (default: 3, max: 5)."
                    ),
                },
                required=["event_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_impact_chain",
            description="""Trace the downstream impacts and effects of this event.
Use to answer 'What happens next?' or 'Who's affected?'.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "event_id": types.Schema(
                        type=types.Type.STRING,
                        description="The event ID to trace forward from."
                    ),
                    "max_depth": types.Schema(
                        type=types.Type.NUMBER,
                        description="Steps forward to trace (default: 3, max: 5)."
                    ),
                },
                required=["event_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_entity_events",
            description="""Get recent events involving a specific entity.
Use to answer 'What else has [entity] been doing lately?'.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "entity_name": types.Schema(
                        type=types.Type.STRING,
                        description="The entity name."
                    ),
                    "limit": types.Schema(
                        type=types.Type.NUMBER,
                        description="Max events to return (default: 5, max: 10)."
                    ),
                },
                required=["entity_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_event_details",
            description="""Get the full details (title, summary, severity, sources) of a specific event by its ID.
Use this when you find an event ID in a graph traversal (causal chain, impact chain) and need to read the actual content.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "event_id": types.Schema(
                        type=types.Type.STRING,
                        description="The ID of the event to retrieve."
                    ),
                },
                required=["event_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="resolve_entity",
            description="""Resolve an entity name (e.g. 'Washington', 'CCP', 'TSMC') to its canonical ID and type.
Use this as the entry point when you encounter a new actor in search results and want to query the graph.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "name": types.Schema(
                        type=types.Type.STRING,
                        description="The name, slug, or alias of the entity."
                    ),
                },
                required=["name"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_event_entities",
            description="""Get all entities (countries, companies, leaders) involved in or affected by a specific event.
Use to answer 'Who are the primary actors here?' for a specific event ID.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "event_id": types.Schema(
                        type=types.Type.STRING,
                        description="The ID of the event."
                    ),
                },
                required=["event_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_entities",
            description="""Search for entities by name, role, or sector using keyword matching.
Use this to find actors when you don't have an exact name (e.g., 'semiconductor' or 'iranian leader').""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query (name, industry, or role)."
                    ),
                    "limit": types.Schema(
                        type=types.Type.NUMBER,
                        description="Max results (default: 5)."
                    ),
                },
                required=["query"],
            ),
        ),
    ]


# =============================================================================
# TAVILY SEARCH TOOL
# =============================================================================

def get_tavily_search_tool() -> types.Tool:
    """
    Get the Tavily search tool definition for use by any agent.
    
    Returns a FunctionDeclaration that agents can include in their tools.
    """
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_web",
                description="""Search the web for current information about geopolitical events, actors, or context.

WHEN TO SEARCH:
- Recent events you're uncertain about (verify what actually happened)
- Current status of developing situations (e.g., "Iran protests current death toll 2026")
- Actor context for recent appointments or changes (e.g., "Ahmed al-Sharaa Syria current role")
- Significance verification (e.g., "Trump Greenland bid geopolitical implications")

HOW MANY SEARCHES:
- Editor: 0-3 searches total (only for key kill/publish decisions when genuinely uncertain)
- Analyst: 3-8 searches per cluster (verify actors, context, recent developments)
- Use judiciously - each search adds latency and cost

GOOD SEARCH QUERIES (specific, dated, focused):
✅ "Trump Greenland acquisition bid January 2026"
✅ "Iran protests death toll 2026 current"
✅ "Syria ISIS prison breaks December 2025"
✅ "Ahmed al-Sharaa Syria new government"
✅ "Venezuela Maduro January 2026"

BAD SEARCH QUERIES (vague, too broad):
❌ "Trump foreign policy" (too vague)
❌ "Iran context" (too broad)
❌ "Syria crisis" (not specific enough)
❌ "What is happening in Middle East" (too general)

Be specific: include actor names, event keywords, and dates.""",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Specific search query with dates and keywords (e.g., 'Trump Greenland January 2026', 'Iran protests death toll 2026')"
                        ),
                    },
                    required=["query"],
                ),
            ),
        ]
    )


# =============================================================================
# MEMORY TOOLS
# =============================================================================

def get_memory_tools() -> list[types.FunctionDeclaration]:
    """
    Get the set of memory-retrieval tools.
    
    Allows agents to recall specific details from past editions.
    """
    return [
        types.FunctionDeclaration(
            name="read_past_briefing",
            description="""Read the full text (Headline + Rationale/Summary) of a past briefing story or the high-level summary.
Use this when you see a relevant connection in the 'PREVIOUS COVERAGE INDEX' and need more detail to build continuity.""",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "run_id": types.Schema(
                        type=types.Type.STRING,
                        description="The Run ID of the past briefing (e.g., 'briefing_20260125_110850')."
                    ),
                    "section_id": types.Schema(
                        type=types.Type.STRING,
                        description="Optional: The cluster ID/source_cluster_id of the story to read. If omitted, returns the Hub/Arc summary."
                    ),
                },
                required=["run_id"],
            ),
        ),
    ]

def execute_tavily_search(query: str) -> str:
    """
    Execute a Tavily search and return formatted results.
    
    Also stores full results in global storage for CoVe verification.
    
    Args:
        query: Search query string
        
    Returns:
        Formatted string with search results or error message
    """
    global _search_results
    from config import get_config
    
    config = get_config()
    
    if not config.tavily_api_key:
        return "Tavily API key not configured. Using available context only."
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=config.tavily_api_key)
        
        # Use default parameters for flexibility
        response = client.search(query=query)
        
        # Format results for the model AND store full content
        formatted_results = []
        for result in response.get("results", [])[:5]:  # Top 5 results
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', 'No URL')
            score = result.get('score', 0.0)
            
            # Extract publisher from URL domain
            publisher = None
            if url and url != 'No URL':
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    # Clean up domain (remove www., get main part)
                    publisher = domain.replace('www.', '').split('.')[0].title()
                except Exception:
                    publisher = None
            
            # Store full result for verification
            search_result = SearchResult(
                query=query,
                title=title,
                url=url,
                content=content,
                publisher=publisher,
                score=score,
            )
            _search_results.append(search_result)
            
            # Format for LLM
            formatted_results.append(
                f"- {title}\n"
                f"  {content}\n"
                f"  Source: {url}"
            )
        
        # Log storage for transparency
        if formatted_results:
            logger.debug(f"       📚 Stored {len(formatted_results)} search results for verification")
            return "\n\n".join(formatted_results)
        return "No relevant results found."
        
    except ImportError:
        return "Tavily package not installed. Run: pip install tavily-python"
    except Exception as e:
        logger.warning(f"   ⚠️ Tavily search failed: {e}")
        return f"Search failed: {str(e)}"


def execute_tool_call(name: str, args: dict) -> str:
    """
    Execute a tool call by name.
    
    Dispatches to the appropriate executor based on tool name.
    
    Args:
        name: Tool name (e.g., "search_web", "get_event_graph")
        args: Tool arguments dict
        
    Returns:
        Result as string
    """
    if name == "search_web":
        query = args.get("query", "")
        result = execute_tavily_search(query)
        logger.info(f"       🔍 Search: {query[:60]}...")
        return result
    
    elif name in ["get_event_graph", "get_entity_relationships", "get_causal_chain", "get_impact_chain", "get_entity_events", "get_event_details", "resolve_entity", "get_event_entities", "search_entities"]:
        return _execute_graph_tool(name, args)
    
    elif name == "read_past_briefing":
        run_id = args.get("run_id")
        section_id = args.get("section_id")
        from memory import get_briefing_content
        result = get_briefing_content(run_id, section_id)
        logger.info(f"       📚 Memory lookup: {run_id} ({section_id or 'Summary'})")
        return result
    
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def _execute_graph_tool(name: str, args: dict) -> str:
    """Execute a graph-querying tool using Supabase RPCs."""
    try:
        supabase = get_supabase_client()
        
        if name == "get_event_graph":
            event_id = args.get("event_id")
            include_indirect = args.get("include_indirect", False)
            
            # 1. Get entities involved in this event
            response = supabase.table("event_details").select("node_id").eq("node_id", event_id).execute()
            # In the real system, we'd look up the entities linked to this event.
            # For this port, we'll mimic the Pythia logic by querying linked nodes.
            
            # Since the-briefing might not have the full entity lookup logic yet, 
            # we'll look for edges where this event_id is the source or target.
            edges_query = supabase.from_("edges").select(
                "source:source_id(name, node_type), target:target_id(name, node_type), relation_type, percentage, confidence, polarity"
            ).or_(f"source_id.eq.{event_id},target_id.eq.{event_id}")
            
            edges_res = edges_query.execute()
            edges = edges_res.data
            
            if not edges:
                return f"No direct graph relationships found for event {event_id}."
            
            graph_desc = f"Event Graph for {event_id}:\n"
            for edge in edges:
                src = edge.get("source", {}).get("name", "Unknown")
                tgt = edge.get("target", {}).get("name", "Unknown")
                rel = edge.get("relation_type", "unknown")
                pct = f" [{edge['percentage']}%]" if edge.get("percentage") else ""
                graph_desc += f"- {src} --[{rel}]{pct}--> {tgt}\n"
            
            return graph_desc

        elif name == "get_entity_relationships":
            entity_name = args.get("entity_name")
            limit = min(args.get("limit", 10), 20)
            
            # Resolve entity name to ID first (exact match)
            res = supabase.table("nodes").select("id").ilike("name", entity_name).limit(1).execute()
            if not res.data:
                return f"Entity '{entity_name}' not found in knowledge graph."
            
            entity_id = res.data[0]["id"]
            
            # Get relationships
            edges_res = supabase.from_("edges").select(
                "relation_type, target:target_id(name, node_type), confidence, polarity, percentage"
            ).eq("source_id", entity_id).limit(limit).execute()
            
            if not edges_res.data:
                return f"No relationships found for '{entity_name}'."
            
            rels = [f"- {e.get('relation_type')} {e.get('target', {}).get('name')} (conf: {e.get('confidence', 0)})" for e in edges_res.data]
            return f"Relationships for {entity_name}:\n" + "\n".join(rels)

        elif name == "get_causal_chain":
            event_id = args.get("event_id")
            depth = min(args.get("max_depth", 3), 5)
            
            res = supabase.rpc("get_causal_chain", {"event_id": event_id, "max_depth": depth}).execute()
            if not res.data:
                return "No causal chain data available."
            
            chain = [f"{'  ' * (item['depth']-1)}↳ {item['name']} ({item['relation']})" for item in res.data]
            return "Causal chain:\n" + "\n".join(chain)

        elif name == "get_impact_chain":
            event_id = args.get("event_id")
            depth = min(args.get("max_depth", 3), 5)
            
            res = supabase.rpc("get_impact_chain", {
                "start_node_id": event_id, 
                "max_depth": depth,
                "min_weight": 0.1
            }).execute()
            
            if not res.data:
                return "No impact chain data available."
            
            chain = [f"{'  ' * (item['depth']-1)}→ {item['name']} ({item['node_type']}) [{int(item['cumulative_weight']*100)}% impact]" for item in res.data]
            return "Downstream impacts:\n" + "\n".join(chain)

        elif name == "get_entity_events":
            entity_name = args.get("entity_name")
            limit = min(args.get("limit", 5), 10)
            
            res_node = supabase.table("nodes").select("id").ilike("name", entity_name).limit(1).execute()
            if not res_node.data:
                return f"Entity '{entity_name}' not found."
            
            entity_id = res_node.data[0]["id"]
            res = supabase.rpc("get_entity_events", {"entity_uuid": entity_id, "max_count": limit}).execute()
            
            if not res.data:
                return f"No recent events found for {entity_name}."
            
            events = [f"- {e['title']} ({e['event_timestamp'][:10]})" for e in res.data]
            return f"Recent events for {entity_name}:\n" + "\n".join(events)

        elif name == "get_event_details":
            event_id = args.get("event_id")
            
            # Query from events_with_reactions view (matches aggregate.py)
            res = supabase.table("events_with_reactions").select("*").eq("id", event_id).limit(1).execute()
            
            if not res.data:
                return f"Event '{event_id}' not found."
            
            event = res.data[0]
            summary = event.get("summary", "No summary available.")
            return json.dumps({
                "id": event.get("id"),
                "title": event.get("title"),
                "summary": summary,
                "severity": event.get("severity"),
                "category": event.get("category"),
                "timestamp": event.get("timestamp"),
                "location": event.get("location_name"),
                "sources_count": len(event.get("sources", [])) if isinstance(event.get("sources"), list) else 1
            }, indent=2)

        elif name == "resolve_entity":
            name_val = args.get("name")
            # Use the robust resolve_entity RPC from realpolitik
            res = supabase.rpc("resolve_entity", {"entity_name": name_val}).execute()
            
            if not res.data:
                return f"Entity '{name_val}' not found in knowledge graph."
            
            entity_id = res.data
            # Get basic info for the resolved entity
            node_res = supabase.table("nodes").select("name, node_type, slug").eq("id", entity_id).single().execute()
            node = node_res.data
            return json.dumps({
                "id": entity_id,
                "name": node.get("name"),
                "type": node.get("node_type"),
                "slug": node.get("slug")
            }, indent=2)

        elif name == "get_event_entities":
            event_id = args.get("event_id")
            res = supabase.rpc("get_event_entities", {"event_uuid": event_id}).execute()
            
            if not res.data:
                return f"No entities found linked to event {event_id}."
            
            entities = [f"- {e['name']} ({e['node_type']}) - relation: {e['relation_type']}" for e in res.data]
            return f"Entities involved in event {event_id}:\n" + "\n".join(entities)

        elif name == "search_entities":
            query = args.get("query")
            limit = min(args.get("limit", 5), 10)
            
            # Use ilike on nodes table for a simple keyword search if hybrid_search is unavailable or needs embedding
            # OR we can try to use the search_vector if it's indexed
            res = supabase.table("nodes") \
                .select("id, name, node_type, slug") \
                .ilike("name", f"%{query}%") \
                .neq("node_type", "event") \
                .limit(limit) \
                .execute()
            
            if not res.data:
                return f"No entities found matching '{query}'."
            
            results = [f"- {n['name']} ({n['node_type']}) [slug: {n['slug']}] [id: {n['id']}]" for n in res.data]
            return f"Search results for '{query}':\n" + "\n".join(results)

    except Exception as e:
        logger.warning(f"   ⚠️ Graph tool '{name}' failed: {e}")
        return f"Tool execution failed: {str(e)}"


def save_search_results_log(output_dir: str) -> str:
    """
    Save all search results to a JSON file for transparency.
    
    Args:
        output_dir: Directory to save the log file
        
    Returns:
        Path to saved file
    """
    import os
    
    filepath = os.path.join(output_dir, "agent_outputs", "00_search_results.json")
    
    results_data = {
        "total_searches": len(_search_results),
        "results": [r.to_dict() for r in _search_results],
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"   📚 Saved {len(_search_results)} search results to {filepath}")
    return filepath
