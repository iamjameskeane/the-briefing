"""
The Briefing - Shared Tools

Common tool definitions and executors used across multiple agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from google.genai import types

from utils import logger


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
        name: Tool name (e.g., "search_web")
        args: Tool arguments dict
        
    Returns:
        Result as string
    """
    if name == "search_web":
        query = args.get("query", "")
        result = execute_tavily_search(query)
        logger.info(f"       🔍 Search: {query[:60]}...")
        return result
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


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
