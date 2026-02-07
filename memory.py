"""
Narrative Memory System - Indexing and Retrieval

Manages outputs/briefing_index.json for high-level continuity awareness
and provides on-demand retrieval of past briefing content.
"""

import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from utils import logger
from state import DocumentSkeleton

INDEX_FILE = "outputs/briefing_index.json"

class BriefingIndex:
    """
    Manages a lightweight index of all published briefings.
    Used by agents to see a 'menu' of past coverage.
    """
    
    def __init__(self, index_file: str = INDEX_FILE):
        self.index_path = Path(index_file)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create empty index if it doesn't exist."""
        if not self.index_path.exists():
            with open(self.index_path, "w") as f:
                json.dump([], f)

    def load_index(self) -> list[dict]:
        """Load and return the list of past briefing metadata."""
        try:
            with open(self.index_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load briefing index: {e}")
            return []

    def save_index(self, index: list[dict]):
        """Save the index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def update_index(self, run_id: str, date: str, skeleton: DocumentSkeleton):
        """
        Add a new briefing to the index based on its final skeleton.
        
        Args:
            run_id: Unique ID for the run
            date: ISO date of the run
            skeleton: The final DocumentSkeleton produced by the pipeline
        """
        index = self.load_index()
        
        # Check if already indexed
        if any(item["run_id"] == run_id for item in index):
            logger.info(f"Run {run_id} already in index, skipping update.")
            return

        # Build entry
        entry = {
            "run_id": run_id,
            "date": date,
            "hub": skeleton.hub_mechanism if skeleton.organizing_principle == "THEMATIC" else "REGIONAL",
            "arc": skeleton.narrative_arc,
            "headlines": {
                skeleton.featured.source_cluster_id: skeleton.featured.headline
            }
        }
        
        # Add section headlines
        for section in skeleton.sections:
            entry["headlines"][section.source_cluster_id] = section.headline
            
        # Add to index
        index.append(entry)
        
        # Sort by date (newest first)
        index.sort(key=lambda x: x["date"], reverse=True)
        
        # Keep index manageable (e.g., last 52 weeks)
        if len(index) > 52:
            index = index[:52]
            
        self.save_index(index)
        logger.info(f"   💾 Updated briefing index with run {run_id}")

def get_briefing_content(run_id: str, cluster_id: Optional[str] = None) -> str:
    """
    Retrieve specific text content from a past briefing.
    
    Args:
        run_id: The run ID to look up (e.g., briefing_20260125_110850)
        cluster_id: Optional cluster ID to get a specific story. 
                   If None, returns the Hub/Arc/High-level summary.
    
    Returns:
        String content of the requested section or high-level summary.
    """
    outputs_dir = Path("outputs")
    run_dir = outputs_dir / run_id
    skeleton_path = run_dir / "agent_outputs" / "02b_document_skeleton.json"
    
    if not skeleton_path.exists():
        return f"Error: Skeleton not found for run {run_id}"
    
    try:
        with open(skeleton_path, "r") as f:
            data = json.load(f)
            
        # 1. High-level request (Hub/Arc)
        if not cluster_id:
            hub = data.get("hub_mechanism", "N/A")
            exp = data.get("hub_explanation", "N/A")
            arc = data.get("narrative_arc", "N/A")
            return f"Hub: {hub}\nExplanation: {exp}\nNarrative Arc: {arc}"
            
        # 2. Specific story request
        # Check featured
        featured = data.get("featured", {})
        if featured.get("source_cluster_id") == cluster_id:
            return f"HEADLINE: {featured.get('headline')}\nANGLE: {featured.get('angle')}\nRATIONALE: {featured.get('rationale')}"
            
        # Check sections
        for section in data.get("sections", []):
            if section.get("source_cluster_id") == cluster_id:
                return f"HEADLINE: {section.get('headline')}\nTREATMENT: {section.get('treatment')}\nRATIONALE: {section.get('rationale')}"
                
        # Check quick hits
        for qh in data.get("quick_hits", []):
            if qh.get("region") == cluster_id or qh.get("event_id") == cluster_id:
                return f"HEADLINE: {qh.get('headline')}\nCONTENT: {qh.get('content')}"
                
        return f"Error: Story ID '{cluster_id}' not found in run {run_id}"
        
    except Exception as e:
        return f"Error retrieving content: {e}"

def format_index_for_prompt(index_data: list[dict]) -> str:
    """Format the index list into a string for an agent's prompt."""
    if not index_data:
        return "No previous editions indexed."
        
    lines = ["## 📚 PREVIOUS COVERAGE INDEX"]
    lines.append("Use the 'read_past_briefing' tool if a current cluster seems related to these stories.")
    lines.append("")
    
    for entry in index_data:
        date = entry.get("date", "Unknown")
        run_id = entry.get("run_id", "Unknown")
        hub = entry.get("hub", "REGIONAL")
        lines.append(f"[{date}] Run ID: {run_id}")
        lines.append(f"Hub: {hub}")
        lines.append("Stories:")
        for cid, headline in entry.get("headlines", {}).items():
            lines.append(f"  - {headline} ({cid})")
        lines.append("")
        
    return "\n".join(lines)
