"""
The Briefing - Semantic Clustering Phase

Uses embeddings and HDBSCAN to cluster related events into storylines.
Handles black swan events (high-severity outliers) separately.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

from google import genai
from google.genai import types
import numpy as np
from hdbscan import HDBSCAN
from pydantic import BaseModel, Field

from utils import with_retry, logger, log_model_config
from config import get_config


TrendDirection = Literal["ESCALATING", "DE_ESCALATING", "STABLE", "VOLATILE"]


@dataclass
class Storyline:
    """A cluster of related events that form a storyline."""
    cluster_id: int
    events: list[dict]
    event_ids: list[str]
    primary_region: str
    avg_severity: float
    
    def to_dict(self) -> dict:
        return {
            "cluster_id": int(self.cluster_id),
            "event_count": len(self.events),
            "event_ids": self.event_ids,
            "primary_region": self.primary_region,
            "avg_severity": float(round(self.avg_severity, 1)),
        }


@dataclass
class RegionAnalysis:
    """Analysis for a single region."""
    region: str
    storylines: list[Storyline]
    trend: TrendDirection
    total_events: int
    high_severity_count: int
    
    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "trend": self.trend,
            "total_events": int(self.total_events),
            "high_severity_count": int(self.high_severity_count),
            "storyline_count": len(self.storylines),
            "storylines": [s.to_dict() for s in self.storylines],
        }


@dataclass
class CrossRegionalConnection:
    """Connection between events in different regions."""
    event_ids: list[str]
    regions: list[str]
    connection_type: str  # "same_actor", "causal_chain", "coordinated_timing"
    explanation: str
    
    def to_dict(self) -> dict:
        return {
            "event_ids": self.event_ids,
            "regions": self.regions,
            "connection_type": self.connection_type,
            "explanation": self.explanation,
        }


@dataclass
class ThematicCluster:
    """
    A theme-based cluster that may span multiple regions.
    
    This is the key to Hub-and-Spoke organization:
    - If a thematic cluster touches 3+ regions, it's a potential "Hub"
    - The regions become "Spokes" showing regional manifestations
    """
    cluster_id: int
    theme_label: str  # LLM-generated label like "US Policy Shifts", "Energy Crisis"
    events: list[dict]
    event_ids: list[str]
    regions_touched: list[str]  # Which regions this theme spans
    region_event_counts: dict[str, int]  # region -> count of events
    avg_severity: float
    is_hub_candidate: bool  # True if touches 3+ regions with significant events
    
    def to_dict(self) -> dict:
        return {
            "cluster_id": int(self.cluster_id),
            "theme_label": self.theme_label,
            "event_count": len(self.events),
            "event_ids": self.event_ids,
            "events": self.events,  # Include full event objects for Architect
            "regions_touched": self.regions_touched,
            "region_event_counts": self.region_event_counts,
            "avg_severity": float(round(self.avg_severity, 1)),
            "is_hub_candidate": self.is_hub_candidate,
        }


@dataclass
class ClusterAnalysis:
    """Complete output of Phase 2: Clustered events ready for generation."""
    regions: dict[str, RegionAnalysis] = field(default_factory=dict)
    black_swan_events: list[dict] = field(default_factory=list)
    emerging_situations: list[dict] = field(default_factory=list)
    cross_regional_connections: list[CrossRegionalConnection] = field(default_factory=list)
    
    # Thematic clusters that span regions (for Hub-and-Spoke)
    thematic_clusters: list[ThematicCluster] = field(default_factory=list)
    hub_candidates: list[ThematicCluster] = field(default_factory=list)  # Themes touching 3+ regions
    recommended_organization: str = "REGIONAL"  # REGIONAL, THEMATIC, or HYBRID
    
    def to_dict(self) -> dict:
        def _safe_severity(s):
            """Convert severity to native Python int."""
            if s is None:
                return 0
            return int(s)
        
        return {
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "black_swan_count": len(self.black_swan_events),
            "black_swan_events": [
                {
                    "id": e.get("id"),
                    "title": e.get("title"),
                    "severity": _safe_severity(e.get("severity")),
                }
                for e in self.black_swan_events
            ],
            "emerging_situations_count": len(self.emerging_situations),
            "cross_regional_connections": [c.to_dict() for c in self.cross_regional_connections],
            # NEW thematic fields
            "thematic_clusters": [t.to_dict() for t in self.thematic_clusters],
            "hub_candidates": [h.to_dict() for h in self.hub_candidates],
            "recommended_organization": self.recommended_organization,
        }


def get_genai_client() -> genai.Client:
    """Configure and return Gemini client for embeddings."""
    config = get_config()
    if not config.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=config.gemini_api_key)


def create_event_text(event: dict) -> str:
    """
    Create text representation of an event for embedding.
    
    Combines title, summary, location, and category for rich semantic representation.
    """
    parts = []
    
    # Title is most important
    title = event.get("title", "")
    if title:
        parts.append(title)
    
    # Summary provides context
    summary = event.get("summary", "")
    if summary:
        parts.append(summary)
    
    # Location helps with geographic clustering
    location = event.get("location_name", "")
    if location:
        parts.append(f"Location: {location}")
    
    # Category for thematic clustering
    category = event.get("category", "")
    if category:
        parts.append(f"Category: {category}")
    
    return " | ".join(parts)


@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
def _embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with retry logic."""
    config = get_config()
    result = client.models.embed_content(
        model=config.models.embedding,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="CLUSTERING",
            output_dimensionality=config.embedding_dimensions,
        ),
    )
    return [e.values for e in result.embeddings]


def embed_events(events: list[dict]) -> np.ndarray:
    """
    Generate embeddings for a list of events using Gemini.
    
    Args:
        events: List of event dictionaries.
    
    Returns:
        NumPy array of embeddings, shape (n_events, 768).
    """
    if not events:
        return np.array([])
    
    client = get_genai_client()
    
    # Create text representations
    texts = [create_event_text(event) for event in events]
    
    logger.info(f"Generating embeddings for {len(texts)} events...")
    
    # Batch embeddings (Gemini limit is 100 per batch)
    config = get_config()
    batch_size = config.embedding_batch_size
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)...")
        
        batch_embeddings = _embed_batch(client, batch)
        all_embeddings.extend(batch_embeddings)
    
    # Normalize embeddings (required for reduced dimensions per Gemini docs)
    embedding_array = np.array(all_embeddings)
    norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embedding_array / norms
    
    return normalized


def cluster_events_hdbscan(
    events: list[dict], 
    embeddings: np.ndarray
) -> tuple[dict[int, list[dict]], list[dict]]:
    """
    Cluster events using HDBSCAN.
    
    HDBSCAN is preferred over K-Means because:
    - Doesn't require specifying number of clusters
    - Handles noise (outlier events that don't fit storylines)
    - Finds clusters of varying density
    
    Args:
        events: List of event dictionaries.
        embeddings: NumPy array of embeddings.
    
    Returns:
        Tuple of (clusters_dict, black_swan_events)
        - clusters_dict: Maps cluster_id -> list of events
        - black_swan_events: High-severity events marked as noise
    """
    config = get_config()
    if len(events) < config.hdbscan_min_cluster_size:
        # Not enough events to cluster
        return {}, []
    
    # HDBSCAN with cosine distance for semantic similarity
    clusterer = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric='euclidean',  # Use euclidean on normalized vectors (equivalent to cosine)
        cluster_selection_method='eom',  # Excess of Mass (better for varying sizes)
    )
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms
    
    labels = clusterer.fit_predict(normalized)
    
    # Group events by cluster
    clusters: dict[int, list[dict]] = {}
    black_swan_events: list[dict] = []
    noise_count = 0
    
    for idx, label in enumerate(labels):
        event = events[idx]
        severity = event.get("severity", 0)
        
        if label == -1:
            noise_count += 1
            # BLACK SWAN OVERRIDE: High-severity events NEVER get discarded
            if severity >= config.black_swan_threshold:
                black_swan_events.append(event)
            continue
        
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(event)
    
    n_clusters = len(clusters)
    print(f"   📊 HDBSCAN: {n_clusters} clusters, {noise_count} noise, {len(black_swan_events)} black swan")
    
    return clusters, black_swan_events


def cluster_globally_by_theme(
    events: list[dict],
    embeddings: np.ndarray,
) -> tuple[list[ThematicCluster], list[dict]]:
    """
    Cluster ALL events globally by theme, ignoring region boundaries.
    
    This is the PRIMARY clustering approach:
    - Find themes that span multiple regions
    - Identify potential "Hub" stories
    - Extract black swan events (high-severity noise)
    
    Args:
        events: All events for the week
        embeddings: Pre-computed embeddings for all events
        
    Returns:
        Tuple of (thematic_clusters, black_swan_events)
    """
    config = get_config()
    if len(events) < config.hdbscan_min_cluster_size:
        return [], []
    
    logger.info("🌍 Performing GLOBAL thematic clustering...")
    
    # Use slightly larger min_cluster_size for global to find broader themes
    clusterer = HDBSCAN(
        min_cluster_size=max(config.hdbscan_min_cluster_size, 4),
        min_samples=2,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    
    labels = clusterer.fit_predict(normalized)
    
    # Group events by cluster, extract black swans from noise
    cluster_events: dict[int, list[dict]] = {}
    black_swan_events: list[dict] = []
    noise_count = 0
    
    for idx, label in enumerate(labels):
        event = events[idx]
        severity = event.get("severity", 0)
        
        if label == -1:
            noise_count += 1
            # BLACK SWAN: High-severity events in noise are significant outliers
            if severity >= config.black_swan_threshold:
                black_swan_events.append(event)
            continue
            
        if label not in cluster_events:
            cluster_events[label] = []
        cluster_events[label].append(event)
    
    logger.info(f"   📊 HDBSCAN: {len(cluster_events)} clusters, {noise_count} noise, {len(black_swan_events)} black swan")
    
    # Build ThematicCluster objects
    thematic_clusters = []
    for cluster_id, cluster_evts in cluster_events.items():
        if len(cluster_evts) < 2:
            continue
            
        # Count regions touched
        region_counts: dict[str, int] = {}
        for evt in cluster_evts:
            region = evt.get("region", "OTHER")
            region_counts[region] = region_counts.get(region, 0) + 1
        
        regions_touched = list(region_counts.keys())
        
        # Calculate average severity
        severities = [e.get("severity", 5) for e in cluster_evts]
        avg_severity = sum(severities) / len(severities)
        
        # Hub candidate if either:
        # Option 1: Multi-region story (3+ regions, 2+ events/region, severity >= 6)
        # Option 2: Major single-region story (20+ events, severity >= 7)
        # Option 3: Critical single-region story (5+ events, severity >= 9)
        is_multi_region_hub = (
            len(regions_touched) >= 3 and 
            len(cluster_evts) >= len(regions_touched) * 2 and
            avg_severity >= 6
        )
        is_major_single_region = (
            len(cluster_evts) >= 20 and
            avg_severity >= 7
        )
        is_critical_event = (
            len(cluster_evts) >= 5 and
            avg_severity >= 9
        )
        
        is_hub = is_multi_region_hub or is_major_single_region or is_critical_event
        
        thematic_clusters.append(ThematicCluster(
            cluster_id=cluster_id,
            theme_label="",  # Will be labeled by LLM later
            events=cluster_evts,
            event_ids=[e.get("id", "") for e in cluster_evts],
            regions_touched=regions_touched,
            region_event_counts=region_counts,
            avg_severity=avg_severity,
            is_hub_candidate=is_hub,
        ))
    
    # Sort by IMPACT (event count × average severity), not just severity
    # This ensures major stories (68 events) beat smaller high-severity clusters
    thematic_clusters.sort(key=lambda x: -(len(x.events) * x.avg_severity))
    
    return thematic_clusters, black_swan_events


def label_thematic_clusters(clusters: list[ThematicCluster]) -> list[ThematicCluster]:
    """
    Use LLM to generate descriptive labels for thematic clusters.
    
    Args:
        clusters: ThematicCluster objects without labels
        
    Returns:
        Same clusters with theme_label populated
    """
    if not clusters:
        return clusters
    
    # Only label top 5 clusters to save API calls
    to_label = clusters[:5]
    
    client = get_genai_client()
    config = get_config()
    
    for cluster in to_label:
        # Build a summary of the cluster
        titles = [e.get("title", "")[:100] for e in cluster.events[:5]]
        regions = ", ".join(cluster.regions_touched)
        
        prompt = f"""Based on these related events spanning {regions}:

{chr(10).join(f"- {t}" for t in titles)}

Generate a 2-4 word thematic label that captures what connects them.
Examples: "US Policy Shifts", "Energy Crisis", "Trade War Escalation", "Democratic Backsliding"

Respond with ONLY the label, nothing else."""

        logger.info(f"Labeling cluster {cluster.cluster_id}: {len(cluster.events)} events, {len(titles)} titles, {len(prompt)} chars")
        logger.debug(f"Prompt: {prompt[:500]}")

        try:
            gen_config = types.GenerateContentConfig()
            
            # Only set temperature if explicitly configured
            if config.theme_temperature is not None:
                gen_config.temperature = config.theme_temperature
            
            # Only set max_output_tokens if explicitly configured
            if config.theme_max_output_tokens is not None:
                gen_config.max_output_tokens = config.theme_max_output_tokens
            
            # Log config for debugging
            log_model_config(f"ThemeLabeling-Cluster{cluster.cluster_id}", config.models.theme, gen_config)
            
            response = client.models.generate_content(
                model=config.models.theme,
                contents=[prompt],
                config=gen_config,
            )
            
            logger.debug(f"Response object: {response}")
            if hasattr(response, 'usage_metadata'):
                logger.info(f"Usage: {response.usage_metadata}")
            # Handle None or empty response
            if response.text:
                cluster.theme_label = response.text.strip().strip('"')
                logger.info(f"✅ Cluster {cluster.cluster_id} labeled: {cluster.theme_label}")
            else:
                # Log why response might be empty
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                    logger.warning(f"Empty response for cluster {cluster.cluster_id}, finish_reason: {finish_reason}")
                    logger.warning(f"Full candidate: {candidate}")
                    
                    # Try to extract from parts directly
                    if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                        logger.info(f"Parts found: {len(candidate.content.parts)}")
                        for i, part in enumerate(candidate.content.parts):
                            logger.info(f"Part {i}: {part}")
                            if hasattr(part, 'text') and part.text:
                                cluster.theme_label = part.text.strip().strip('"')
                                logger.info(f"✅ Extracted from part: {cluster.theme_label}")
                                break
                
                # If still no label, fallback to first event title words
                if not cluster.theme_label:
                    first_title = titles[0] if titles else "Unknown"
                    cluster.theme_label = " ".join(first_title.split()[:3])
        except Exception as e:
            # Fallback to generic label
            logger.warning(f"Failed to label cluster: {e}")
            cluster.theme_label = f"Theme {cluster.cluster_id}"
    
    return clusters


def determine_recommended_organization(
    thematic_clusters: list[ThematicCluster],
    total_events: int,
) -> str:
    """
    Recommend REGIONAL, THEMATIC, or HYBRID based on cluster analysis.
    
    Logic:
    - If strongest theme touches 60%+ of high-priority events → THEMATIC
    - If hub candidates exist but <60% coverage → HYBRID
    - Otherwise → REGIONAL
    """
    if not thematic_clusters:
        return "REGIONAL"
    
    hub_candidates = [c for c in thematic_clusters if c.is_hub_candidate]
    
    if not hub_candidates:
        return "REGIONAL"
    
    # Check if top hub covers 60%+ of significant events
    top_hub = hub_candidates[0]
    high_sev_events = sum(
        1 for e in top_hub.events if e.get("severity", 0) >= 7
    )
    
    # If top theme has 60%+ of its events as high-severity and spans 4+ regions
    if len(top_hub.regions_touched) >= 4 and top_hub.avg_severity >= 7:
        return "THEMATIC"
    
    # If multiple hub candidates exist
    if len(hub_candidates) >= 2:
        return "HYBRID"
    
    # Single hub but not dominant
    if len(hub_candidates) == 1 and len(top_hub.regions_touched) >= 3:
        return "HYBRID"
    
    return "REGIONAL"


def determine_trend(events: list[dict]) -> TrendDirection:
    """
    Determine trend direction for a set of events.
    
    Based on:
    - Severity distribution
    - Event count (high volume = volatile)
    - Presence of very high severity events
    """
    if not events:
        return "STABLE"
    
    severities = [e.get("severity", 5) for e in events]
    avg_severity = sum(severities) / len(severities)
    max_severity = max(severities)
    high_sev_count = sum(1 for s in severities if s >= 7)
    
    # Volatile: high volume and mixed severity
    if len(events) >= 10 and len(set(severities)) >= 4:
        return "VOLATILE"
    
    # Escalating: high average severity or multiple high-severity events
    if avg_severity >= 6.5 or high_sev_count >= 3 or max_severity >= 9:
        return "ESCALATING"
    
    # De-escalating: low average severity
    if avg_severity <= 4.0 and max_severity <= 6:
        return "DE_ESCALATING"
    
    return "STABLE"


def cluster_region(
    region: str, 
    events: list[dict]
) -> tuple[RegionAnalysis, list[dict]]:
    """
    Cluster events for a single region.
    
    Args:
        region: Region name.
        events: Events in this region.
    
    Returns:
        Tuple of (RegionAnalysis with storylines and trend, black_swan_events).
    """
    if not events:
        return RegionAnalysis(
            region=region,
            storylines=[],
            trend="STABLE",
            total_events=0,
            high_severity_count=0,
        ), []
    
    high_sev = sum(1 for e in events if e.get("severity", 0) >= 7)
    config = get_config()
    
    # If too few events for clustering, treat as single storyline
    if len(events) < config.hdbscan_min_cluster_size:
        storyline = Storyline(
            cluster_id=0,
            events=events,
            event_ids=[e.get("id", "") for e in events],
            primary_region=region,
            avg_severity=sum(e.get("severity", 5) for e in events) / len(events),
        )
        return RegionAnalysis(
            region=region,
            storylines=[storyline],
            trend=determine_trend(events),
            total_events=len(events),
            high_severity_count=high_sev,
        ), []
    
    # Embed and cluster (only once per region)
    embeddings = embed_events(events)
    clusters, black_swan = cluster_events_hdbscan(events, embeddings)
    
    # Convert clusters to storylines
    storylines = []
    for cluster_id, cluster_events in clusters.items():
        severities = [e.get("severity", 5) for e in cluster_events]
        storyline = Storyline(
            cluster_id=cluster_id,
            events=cluster_events,
            event_ids=[e.get("id", "") for e in cluster_events],
            primary_region=region,
            avg_severity=sum(severities) / len(severities),
        )
        storylines.append(storyline)
    
    # Sort storylines by average severity (most important first)
    storylines.sort(key=lambda s: -s.avg_severity)
    
    return RegionAnalysis(
        region=region,
        storylines=storylines,
        trend=determine_trend(events),
        total_events=len(events),
        high_severity_count=high_sev,
    ), black_swan


def identify_emerging_situations(
    region_analyses: dict[str, RegionAnalysis],
    all_events: list[dict],
) -> list[dict]:
    """
    Identify emerging situations that warrant monitoring.
    
    Criteria:
    - Fewer than 5 events (not yet a major storyline)
    - At least severity 6 event present
    - Not part of a large cluster
    """
    emerging = []
    
    # Find isolated high-severity events that aren't in large clusters
    clustered_ids = set()
    for analysis in region_analyses.values():
        for storyline in analysis.storylines:
            if len(storyline.events) >= 5:
                clustered_ids.update(storyline.event_ids)
    
    # Check for emerging patterns
    for event in all_events:
        event_id = event.get("id", "")
        severity = event.get("severity", 0)
        
        if event_id not in clustered_ids and severity >= 6:
            # This is a significant but isolated event
            emerging.append({
                "id": event_id,
                "title": event.get("title", ""),
                "region": event.get("region", "OTHER"),
                "severity": severity,
                "location": event.get("location_name", ""),
            })
    
    # Limit to top 5 by severity
    emerging.sort(key=lambda e: -e.get("severity", 0))
    return emerging[:5]


# Known actors that operate across regions
CROSS_REGIONAL_ACTORS = [
    "russia", "russian", "moscow", "putin", "kremlin",
    "china", "chinese", "beijing", "pla",
    "iran", "iranian", "tehran", "irgc",
    "united states", "us ", "american", "washington", "pentagon",
    "nato", "eu ", "european union",
    "houthi", "hezbollah", "hamas",
    "opec", "brics",
]

# Keywords indicating causal chains
CAUSAL_KEYWORDS = [
    "supply chain", "shipping", "trade route", "energy",
    "oil", "gas", "commodity", "sanctions", "tariff",
    "currency", "inflation", "migration", "refugee",
]


def identify_cross_regional_connections(
    events: list[dict],
    by_region: dict[str, list[dict]],
    embeddings: np.ndarray | None = None,
) -> list[CrossRegionalConnection]:
    """
    Identify connections between events in different regions.
    
    Connection types:
    - semantic_similarity: Events with similar embeddings across regions
    - same_actor: Same actor mentioned in events across regions
    - causal_chain: Economic/supply chain effects spanning regions
    
    Args:
        events: All events for the week.
        by_region: Events grouped by region.
        embeddings: Optional global embeddings for semantic similarity detection.
    
    Returns:
        List of CrossRegionalConnection objects.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    connections: list[CrossRegionalConnection] = []
    
    # === Method 1: Semantic Similarity (if embeddings provided) ===
    if embeddings is not None and len(events) > 0:
        # Create event ID to index mapping
        event_to_idx = {e.get("id", ""): i for i, e in enumerate(events)}
        regions = list(by_region.keys())
        
        # Compare events across region pairs
        for i, r1 in enumerate(regions):
            for r2 in regions[i+1:]:
                r1_events = by_region.get(r1, [])
                r2_events = by_region.get(r2, [])
                
                if not r1_events or not r2_events:
                    continue
                
                # Get indices for each region
                r1_indices = [event_to_idx.get(e.get("id", ""), -1) for e in r1_events]
                r2_indices = [event_to_idx.get(e.get("id", ""), -1) for e in r2_events]
                
                # Filter valid indices
                r1_indices = [i for i in r1_indices if i >= 0]
                r2_indices = [i for i in r2_indices if i >= 0]
                
                if not r1_indices or not r2_indices:
                    continue
                
                # Get embeddings for each region
                r1_emb = embeddings[r1_indices]
                r2_emb = embeddings[r2_indices]
                
                # Compute cross-region similarity matrix
                similarities = cosine_similarity(r1_emb, r2_emb)
                
                # Find highly similar pairs (threshold 0.75+)
                SIMILARITY_THRESHOLD = 0.75
                similar_pairs = np.where(similarities > SIMILARITY_THRESHOLD)
                
                for idx1, idx2 in zip(*similar_pairs):
                    r1_event = r1_events[idx1]
                    r2_event = r2_events[idx2]
                    sim_score = similarities[idx1, idx2]
                    
                    connections.append(CrossRegionalConnection(
                        event_ids=[r1_event.get("id", ""), r2_event.get("id", "")],
                        regions=[r1, r2],
                        connection_type="semantic_similarity",
                        explanation=f"Events share similar themes (similarity: {sim_score:.2f}): '{r1_event.get('title', '')[:40]}...' and '{r2_event.get('title', '')[:40]}...'",
                    ))
    
    # === Method 2: Actor-Based Detection (existing logic) ===
    actor_events: dict[str, list[dict]] = {}
    for event in events:
        text = f"{event.get('title', '')} {event.get('summary', '')}".lower()
        for actor in CROSS_REGIONAL_ACTORS:
            if actor in text:
                if actor not in actor_events:
                    actor_events[actor] = []
                actor_events[actor].append(event)
    
    # Find actors that appear in multiple regions
    for actor, actor_evts in actor_events.items():
        regions = set(e.get("region", "OTHER") for e in actor_evts)
        if len(regions) >= 2:
            connections.append(CrossRegionalConnection(
                event_ids=[e.get("id", "") for e in actor_evts[:5]],
                regions=list(regions),
                connection_type="same_actor",
                explanation=f"Actor '{actor}' involved in events across {', '.join(regions)}",
            ))
    
    # === Method 3: Causal Chain Detection ===
    causal_events: list[dict] = []
    for event in events:
        text = f"{event.get('title', '')} {event.get('summary', '')}".lower()
        for keyword in CAUSAL_KEYWORDS:
            if keyword in text:
                causal_events.append(event)
                break
    
    causal_regions = set(e.get("region", "OTHER") for e in causal_events)
    if len(causal_regions) >= 2 and len(causal_events) >= 2:
        connections.append(CrossRegionalConnection(
            event_ids=[e.get("id", "") for e in causal_events[:5]],
            regions=list(causal_regions),
            connection_type="causal_chain",
            explanation=f"Economic/supply chain effects spanning {', '.join(causal_regions)}",
        ))
    
    # Deduplicate by keeping unique connection types per region pair
    seen = set()
    unique_connections = []
    for conn in connections:
        key = (frozenset(conn.regions), conn.connection_type)
        if key not in seen:
            seen.add(key)
            unique_connections.append(conn)
    
    # Sort by connection type priority: semantic > same_actor > causal_chain
    type_priority = {"semantic_similarity": 0, "same_actor": 1, "causal_chain": 2}
    unique_connections.sort(key=lambda c: type_priority.get(c.connection_type, 99))
    
    # Limit to top 5 connections
    return unique_connections[:5]


def cluster_week(
    events: list[dict],
    by_region: dict[str, list[dict]],
) -> ClusterAnalysis:
    """
    Main Phase 2 function: Cluster all events and prepare for generation.
    
    Optimization: Embeds ALL events once globally, then uses those embeddings
    for both per-region clustering and cross-regional connection detection.
    
    Args:
        events: All events for the week.
        by_region: Events grouped by region.
    
    Returns:
        ClusterAnalysis ready for Phase 3 generation.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: THEMATIC CLUSTERING")
    logger.info("=" * 60)
    
    # === Embed ALL events once globally ===
    logger.info("Generating global embeddings for all events...")
    global_embeddings = embed_events(events) if events else np.array([])
    
    # === THEMATIC-FIRST: Cluster ALL events by theme, not region ===
    thematic_clusters: list[ThematicCluster] = []
    hub_candidates: list[ThematicCluster] = []
    all_black_swan: list[dict] = []
    recommended_org = "REGIONAL"
    config = get_config()
    
    if len(events) >= config.hdbscan_min_cluster_size:
        # Global thematic clustering - this is the PRIMARY clustering now
        thematic_clusters, all_black_swan = cluster_globally_by_theme(events, global_embeddings)
        
        # Label top thematic clusters
        if thematic_clusters:
            thematic_clusters = label_thematic_clusters(thematic_clusters)
        
        hub_candidates = [c for c in thematic_clusters if c.is_hub_candidate]
        recommended_org = determine_recommended_organization(thematic_clusters, len(events))
        
        logger.info(f"   📊 Thematic clusters: {len(thematic_clusters)}")
        logger.info(f"   🎯 Hub candidates: {len(hub_candidates)}")
        logger.info(f"   🎯 Recommended organization: {recommended_org}")
    
    # === Build RegionAnalysis from thematic clusters (for backward compatibility) ===
    # Each thematic cluster contributes to its touched regions
    region_analyses: dict[str, RegionAnalysis] = {}
    for region, region_events in by_region.items():
        if not region_events:
            continue
        
        # Find clusters that touch this region
        region_storylines = []
        for tc in thematic_clusters:
            if region in tc.regions_touched:
                # Filter to just this region's events from the cluster
                region_cluster_events = [e for e in tc.events if e.get("region") == region]
                if region_cluster_events:
                    storyline = Storyline(
                        cluster_id=tc.cluster_id,
                        events=region_cluster_events,
                        event_ids=[e.get("id", "") for e in region_cluster_events],
                        primary_region=region,
                        avg_severity=tc.avg_severity,
                    )
                    region_storylines.append(storyline)
        
        # If no storylines from thematic clusters, create one from all region events
        if not region_storylines and region_events:
            region_storylines.append(Storyline(
                    cluster_id=0,
                    events=region_events,
                    event_ids=[e.get("id", "") for e in region_events],
                    primary_region=region,
                    avg_severity=sum(e.get("severity", 5) for e in region_events) / len(region_events),
            ))
        
        high_sev = sum(1 for e in region_events if e.get("severity", 0) >= 7)
        region_analyses[region] = RegionAnalysis(
            region=region,
            storylines=region_storylines,
            trend=determine_trend(region_events),
            total_events=len(region_events),
            high_severity_count=high_sev,
        )
        logger.info(f"   📍 {region}: {len(region_storylines)} storylines from thematic clusters")
    
    # Select featured topic (now based on thematic clusters)
    # Identify emerging situations
    emerging = identify_emerging_situations(region_analyses, events)
    logger.info(f"Emerging situations: {len(emerging)}")
    
    # Cross-regional connections are now implicit in thematic clusters
    cross_regional = identify_cross_regional_connections(events, by_region, global_embeddings)
    logger.info(f"Cross-regional connections: {len(cross_regional)}")
    
    result = ClusterAnalysis(
        regions=region_analyses,
        black_swan_events=all_black_swan,
        emerging_situations=emerging,
        cross_regional_connections=cross_regional,
        # NEW thematic fields
        thematic_clusters=thematic_clusters,
        hub_candidates=hub_candidates,
        recommended_organization=recommended_org,
    )
    
    # Summary
    total_storylines = sum(len(a.storylines) for a in region_analyses.values())
    logger.info(f"Total storylines: {total_storylines}")
    logger.info(f"Black swan events: {len(all_black_swan)}")
    logger.info(f"Cross-regional: {len(cross_regional)}")
    logger.info(f"Thematic clusters: {len(thematic_clusters)}")
    logger.info(f"Hub candidates: {len(hub_candidates)}")
    
    return result


if __name__ == "__main__":
    """Test clustering standalone."""
    import json
    import sys
    from dotenv import load_dotenv
    
    from aggregate import aggregate_week
    
    load_dotenv()
    
    try:
        # Phase 1
        week_data = aggregate_week()
        
        # Phase 2
        clusters = cluster_week(week_data.events, week_data.by_region)
        
        print("\n" + "=" * 60)
        print("✅ Clustering complete!")
        print(json.dumps(clusters.to_dict(), indent=2))
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
