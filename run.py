#!/usr/bin/env python3
"""
The Briefing: Intelligence Analysis Pipeline

Multi-agent editorial pipeline with 8 agents:
- Editor: Editorial research & kill decisions
- Architect: Document skeleton & narrative arc
- Analyst: Constraints-of-Thought, Futures Wheel, ACH reasoning
- Structure: Beat sheets and paragraph plans per archetype
- Writer: Chain of Density prose generation
- Stylist: Voice transformation (Economist/Stratfor style)
- Critic: CoVe verification, dual-track feedback routing
- Assembler: Final document assembly with narrative arc

Key Features:
- Kill authority: PIC Matrix scoring (40-60% of stories killed)
- Hub-spoke organization: Thematic clustering with regional manifestations
- Chain of Density: Iterative information compression
- Chain of Verification: Hallucination detection
- Feedback routing: Content issues → Writer, Style issues → Stylist

Usage:
    python run.py --mode test --dry-run   # Test pipeline 
    python run.py --mode production       # Full quality generation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

# Support direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "the_briefing"

from aggregate import aggregate_week
from agents import (
    run_analyst_agent,
    run_writer_agent,
    AnalystInput,
    WriterInput,
    EditorInput,
    EditorDecisions,
    run_editor_agent,
    ArchitectInput,
    run_architect_agent,
    StructureInput,
    run_structure_agent,
    create_fallback_blueprint,
    StylistInput,
    run_stylist_agent,
    extract_sacred_elements,
    run_content_pipeline,
)
from agents.schemas import (
    AnalystOutput,
    ActorAnalysis,
    ConstraintSet,
    FuturesWheel,
    CompetingHypothesis,
)
from assembly import assemble_document, generate_metadata
from cluster import cluster_week
from config import get_config, reset_config
# Removed constraints.py dependency
from images import generate_briefing_images, insert_images_into_markdown
from state import (
    PipelineState,
    SourceDocument,
    RegionCluster,
    SacredElements,
)
from storage import save_briefing, upload_image_to_r2
from tools import reset_search_results, get_all_search_results, save_search_results_log
from utils import logger, reset_cache, setup_file_logging, close_file_logging
from orchestrator import (
    PipelineOrchestrator,
    create_orchestrator,
    run_with_timeout,
    create_minimal_sacred_elements,
)

if TYPE_CHECKING:
    from google import genai


# =============================================================================
# CHECKPOINTING (delegated to orchestrator)
# =============================================================================


def save_checkpoint(state: PipelineState, phase: str, orchestrator: PipelineOrchestrator | None = None) -> None:
    """Save pipeline state after a phase completes."""
    if orchestrator:
        orchestrator.save_checkpoint(phase)
    else:
        # Fallback for backward compatibility
        from orchestrator import CheckpointManager
        CheckpointManager().save(state, phase)


def clear_checkpoints() -> None:
    """
    Clean up legacy global checkpoint directory.
    
    NOTE: As of 2026-01-24, checkpoints are run-specific and stored in
    outputs/{run_id}/.checkpoints/ instead of the global .checkpoints/ directory.
    This function only cleans up the old global directory for backwards compatibility.
    """
    from pathlib import Path
    
    # Clean up old global .checkpoints/ directory (legacy)
    old_checkpoint_dir = Path(__file__).parent / ".checkpoints"
    if old_checkpoint_dir.exists():
        for f in old_checkpoint_dir.glob("*.json"):
            f.unlink()
        logger.info("   🧹 Cleaned up legacy checkpoint directory")


# =============================================================================
# BANNER
# =============================================================================


def print_banner(mode: str):
    """Print startup banner."""
    print("=" * 60)
    print("📰 THE BRIEFING")
    print("   Multi-Agent Intelligence Analysis Pipeline")
    print("   8 Agents | Editorial Authority | CoVe Verification")
    print("=" * 60)
    print(f"   Mode: {mode.upper()}")
    print(f"   Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)


# =============================================================================
# FALLBACK ANALYST OUTPUT
# =============================================================================


def create_fallback_analyst_output(region: str, events: list[dict]) -> AnalystOutput:
    """Create minimal AnalystOutput when Analyst Agent fails."""
    top_event = max(events, key=lambda e: e.get("severity", 0), default={})
    
    return AnalystOutput(
        region=region,
        geopolitical_archetype="Unknown Pattern",
        archetype_explanation="Analyst agent failed - using event summary fallback.",
        primary_actors=[
            ActorAnalysis(
                actor="Unknown Actor",
                intent="Unable to determine from events.",
                constraints=ConstraintSet(
                    geographic="Not analyzed",
                    economic="Not analyzed",
                    political="Not analyzed",
                ),
                likely_action="Requires further analysis.",
            )
        ],
        futures_wheel=FuturesWheel(
            driver_event=top_event.get("title", "Unknown event"),
            driver_event_id=top_event.get("id", "unknown"),
            first_order="Immediate consequences unclear.",
            second_order="Regional effects unclear.",
            third_order="Long-term structural implications not analyzed.",
        ),
        competing_hypotheses=CompetingHypothesis(
            consensus="Primary assessment unavailable.",
            contrarian="Alternative view unavailable.",
            contradicting_evidence=["Not analyzed."],
            evidence_event_ids=[],
        ),
        pmesii_tags={},
        confidence="LOW",
        confidence_rationale="Fallback mode - analyst agent failed.",
    )


# =============================================================================
# PHASE 1: ANALYST AGENTS (Parallel)
# =============================================================================


async def run_analysts_on_themes(
    state: PipelineState,
    thematic_clusters: list,  # list of ThematicCluster
    orchestrator: PipelineOrchestrator | None = None,
) -> dict[str, AnalystOutput]:
    """
    Run Analyst agents on thematic clusters (not regions).
    
    Analysts work on themes, use function calling to get context.
    """
    from cluster import ThematicCluster
    
    config = get_config()
    
    # Build tasks for parallel execution
    tasks: list[tuple[str, any, AnalystOutput]] = []
    
    # Analyze ALL thematic clusters (HDBSCAN already filtered noise)
    # Let the Architect apply PIC Matrix + Delta Test to kill weak stories
    for cluster in thematic_clusters:
        if len(cluster.events) >= config.min_region_events:
            # Use cluster_id as canonical key (reliable matching)
            cluster_key = f"cluster_{cluster.cluster_id}"
            display_name = cluster.theme_label or cluster_key
            primary_region = cluster.regions_touched[0] if cluster.regions_touched else "OTHER"
            fallback = create_fallback_analyst_output(primary_region, cluster.events)
            
            analyst_input = AnalystInput(
                cluster_id=str(cluster.cluster_id),
                theme_label=cluster.theme_label,
                events=cluster.events,
                regions_touched=cluster.regions_touched,
                is_featured=cluster.is_hub_candidate,
            )
            
            # Key by cluster_id, not theme_label (for reliable lookup)
            tasks.append((cluster_key, run_analyst_agent(analyst_input), fallback))
            logger.info(f"   📝 Queuing analysis: {display_name} ({len(cluster.events)} events, regions: {cluster.regions_touched})")
    
    if not tasks:
        logger.warning("   ⚠️ No clusters meet minimum event threshold")
        return {}
    
    # Use orchestrator for parallel execution with metrics
    if orchestrator:
        results = await orchestrator.run_agents_batch(tasks, "analysts")
    else:
        # Fallback to direct execution
        from orchestrator import run_agents_parallel
        agent_results = await run_agents_parallel(tasks)
        results = {name: r.result for name, r in agent_results.items()}
    
    return results


# Legacy function for backward compatibility
async def run_analysts_parallel(
    state: PipelineState,
    regions_data: dict[str, list[dict]],
) -> dict[str, AnalystOutput]:
    """
    DEPRECATED: Use run_analysts_on_themes instead.
    Kept for backward compatibility.
    """
    logger.warning("   ⚠️ Using legacy regional analyst mode")
    
    async def run_one_analyst(region: str, events: list[dict]) -> tuple[str, AnalystOutput]:
        analyst_input = AnalystInput(
            region=region,
            cluster_id=region,
            theme_label=region,
            events=events,
            regions_touched=[region],
            is_featured=False,
        )
        
        try:
            result = await run_analyst_agent(analyst_input)
            return region, result
        except Exception as e:
            logger.error(f"   ❌ [{region}] Analyst failed: {e}")
            state.add_error(f"Analyst failed for {region}: {e}")
            return region, create_fallback_analyst_output(region, events)
    
    config = get_config()
    tasks = []
    for region, events in regions_data.items():
        if len(events) >= config.min_region_events:
            tasks.append(run_one_analyst(region, events))
    
    results = await asyncio.gather(*tasks)
    return dict(results)


# =============================================================================
# PHASE 2A: EDITOR AGENT (Research & Decisions)
# =============================================================================


async def run_editor(
    state: PipelineState,
    thematic_clusters: list[dict],
    hub_candidates: list[dict],
    recommended_organization: str,
    cross_regional_connections: list[str] | None = None,
    calendar_events: list[dict] | None = None,
    orchestrator: PipelineOrchestrator | None = None,
) -> None:
    """
    Run the Editor agent for editorial research and kill/publish decisions.
    
    Uses deep thinking + Tavily to make informed editorial choices.
    """
    MAX_RETRIES = 3
    
    config = get_config()
    editor_input = EditorInput(
        thematic_clusters=thematic_clusters,
        hub_candidates=hub_candidates,
        total_word_budget=config.total_word_budget,
        recommended_organization=recommended_organization,
        cross_regional_connections=cross_regional_connections or [],
        calendar_events=calendar_events or [],
    )
    
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if orchestrator:
                result = await orchestrator.run_agent(
                    run_editor_agent(editor_input),
                    fallback=None,
                    name="Editor",
                )
                if result is not None:
                    state.editor_decisions = result
                    logger.info(f"   ✅ Editor succeeded on attempt {attempt}")
                    return
                else:
                    raise ValueError("Editor returned None")
            else:
                result = await run_with_timeout(
                    run_editor_agent(editor_input),
                    fallback=None,
                    name="Editor",
                )
                if result.success and result.result is not None:
                    state.editor_decisions = result.result
                    logger.info(f"   ✅ Editor succeeded on attempt {attempt}")
                    return
                else:
                    raise ValueError(result.error or "Editor returned None")
                    
        except Exception as e:
            last_error = str(e)
            logger.warning(f"   ⚠️ Editor attempt {attempt} failed: {last_error}")
            if attempt < MAX_RETRIES:
                logger.info(f"   🔄 Retrying in 2 seconds...")
                await asyncio.sleep(2)
    
    # All retries exhausted - abort
    error_msg = f"Editor failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    logger.error(f"   ❌ CRITICAL: {error_msg}")
    logger.error("   ❌ Aborting pipeline to avoid burning money on bad data.")
    state.add_error(error_msg)
    raise RuntimeError(error_msg)


# =============================================================================
# PHASE 2B: ARCHITECT AGENT (Structure into JSON)
# =============================================================================


async def run_architect(
    state: PipelineState,
    orchestrator: PipelineOrchestrator | None = None,
) -> None:
    """
    Run the Architect agent to structure Editor decisions into DocumentSkeleton JSON.
    
    CRITICAL: Requires Editor to have run successfully first.
    """
    if not state.editor_decisions:
        raise ValueError("Cannot run Architect without Editor decisions")
    
    MAX_RETRIES = 3
    
    config = get_config()
    architect_input = ArchitectInput(
        editor_decisions=state.editor_decisions,
        total_word_budget=config.total_word_budget,
    )
    
    last_error = None
    
    # Get output_dir from orchestrator for immediate reasoning output
    output_dir = orchestrator.output_dir if orchestrator else None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if orchestrator:
                result = await orchestrator.run_agent(
                    run_architect_agent(architect_input, output_dir=output_dir),
                    fallback=None,
                    name="Architect",
                )
                if result is not None:
                    # Unpack tuple: (DocumentSkeleton, reasoning_text)
                    skeleton, reasoning = result
                    normalize_section_groups(skeleton)  # Fix region names -> cluster_ids
                    state.document_skeleton = skeleton
                    state.architect_reasoning = reasoning
                    logger.info(f"   ✅ Architect succeeded on attempt {attempt}")
                    return
                else:
                    raise ValueError("Architect returned None")
            else:
                result = await run_with_timeout(
                    run_architect_agent(architect_input, output_dir=output_dir),
                    fallback=None,
                    name="Architect",
                )
                if result.success and result.result is not None:
                    # Unpack tuple: (DocumentSkeleton, reasoning_text)
                    skeleton, reasoning = result.result
                    normalize_section_groups(skeleton)  # Fix region names -> cluster_ids
                    state.document_skeleton = skeleton
                    state.architect_reasoning = reasoning
                    logger.info(f"   ✅ Architect succeeded on attempt {attempt}")
                    return
                else:
                    raise ValueError(result.error or "Architect returned None")
                    
        except Exception as e:
            last_error = str(e)
            logger.warning(f"   ⚠️ Architect attempt {attempt} failed: {last_error}")
            if attempt < MAX_RETRIES:
                logger.info(f"   🔄 Retrying in 2 seconds...")
                await asyncio.sleep(2)
    
    # All retries exhausted - abort
    error_msg = f"Architect failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    logger.error(f"   ❌ CRITICAL: {error_msg}")
    logger.error("   ❌ Aborting pipeline to avoid burning money on bad data.")
    state.add_error(error_msg)
    raise RuntimeError(error_msg)


# =============================================================================
# SKELETON POST-PROCESSING
# =============================================================================


def normalize_section_groups(skeleton) -> None:
    """
    Fix section_groups.sections to use cluster_ids instead of region names.
    
    The LLM sometimes outputs region names (AMERICAS, EUROPE) instead of 
    cluster IDs (cluster_14, cluster_21) in section_groups.sections.
    This normalizes them to cluster_ids for proper content lookup.
    """
    if not skeleton or not skeleton.section_groups:
        return
    
    # Build mapping from region name -> list of cluster_ids (ordered by appearance)
    # This handles multiple sections with same region (e.g., two MIDDLE_EAST sections)
    region_to_cluster_ids: dict[str, list[str]] = {}
    
    # Add featured first
    if skeleton.featured and skeleton.featured.source_cluster_id:
        region = skeleton.featured.region
        if region not in region_to_cluster_ids:
            region_to_cluster_ids[region] = []
        region_to_cluster_ids[region].append(skeleton.featured.source_cluster_id)
    
    # Then add sections in order
    for section in skeleton.sections:
        if section.source_cluster_id:
            region = section.region
            if region not in region_to_cluster_ids:
                region_to_cluster_ids[region] = []
            region_to_cluster_ids[region].append(section.source_cluster_id)
    
    # Track how many times each region has been used (for handling duplicates)
    region_usage: dict[str, int] = {}
    
    # Normalize each section_group.sections
    for group in skeleton.section_groups:
        normalized_sections = []
        for item in group.sections:
            # If it's already a cluster_id, keep it
            if item.startswith("cluster_"):
                normalized_sections.append(item)
            # Otherwise, it's a region name - look up the cluster_id
            elif item in region_to_cluster_ids:
                cluster_ids = region_to_cluster_ids[item]
                usage_idx = region_usage.get(item, 0)
                if usage_idx < len(cluster_ids):
                    normalized_sections.append(cluster_ids[usage_idx])
                    region_usage[item] = usage_idx + 1
                else:
                    # Ran out of cluster_ids for this region, use last one
                    normalized_sections.append(cluster_ids[-1])
                    logger.warning(f"   ⚠️ Section group has more '{item}' entries than sections exist")
            else:
                # Unknown region, keep as-is and log warning
                logger.warning(f"   ⚠️ Unknown region '{item}' in section_groups, keeping as-is")
                normalized_sections.append(item)
        
        group.sections = normalized_sections
    
    logger.debug(f"   🔧 Normalized section_groups to use cluster_ids")


# =============================================================================
# PHASE 3: STRUCTURE AGENTS (Parallel)
# =============================================================================


async def run_structure_parallel(
    state: PipelineState,
    cluster_events_map: dict[str, list[dict]],
    orchestrator: PipelineOrchestrator | None = None,
) -> None:
    """Run Structure agents for all publishable sections."""
    
    if not state.document_skeleton:
        return
    
    def build_structure_task(section) -> tuple[str, any, any]:
        """Build a task tuple for the orchestrator."""
        from state import SectionBlueprint
        
        region = section.region
        # Use source_cluster_id as THE CANONICAL KEY for all storage (avoid collisions)
        cluster_id = section.source_cluster_id or region
        analyst_output = state.analyst_outputs.get(cluster_id)
        events = cluster_events_map.get(cluster_id, [])
        
        if events:
            logger.debug(f"   📦 [{cluster_id}] Found {len(events)} events (region: {region})")
        
        if not analyst_output:
            state.add_warning(f"No analyst output for {cluster_id} (region: {region})")
            fallback = create_fallback_blueprint(region, create_fallback_analyst_output(region, events), section, log_warning=False)
            # Return immediately-resolved task - KEY BY CLUSTER_ID
            async def noop():
                return fallback
            return cluster_id, noop(), fallback
        
        structure_input = StructureInput(
            region=region,
            analyst_output=analyst_output,
            section_decision=section,
            events=events,
        )
        
        fallback = create_fallback_blueprint(region, analyst_output, section, log_warning=False)
        # KEY BY CLUSTER_ID to avoid collisions (multiple clusters can share same geographic region)
        return cluster_id, run_structure_agent(structure_input), fallback
    
    # Build task list for all sections
    task_tuples: list[tuple[str, any, any]] = []
    
    # Track combined regions (secondary regions to skip)
    combined_secondary_regions = set()
    for section in state.document_skeleton.sections:
        if section.treatment == "COMBINED" and section.combine_with:
            combined_secondary_regions.add(section.combine_with)
    
    # Featured section
    featured = state.document_skeleton.featured
    from state import SectionDecision
    
    featured_decision = SectionDecision(
        region=featured.region,
        source_cluster_id=featured.source_cluster_id,  # Copy cluster ID for lookup
        headline=featured.headline,
        treatment="FULL",
        archetype=featured.story_archetype,
        word_target=featured.word_target,
        position_rationale="Featured story - highest editorial priority",
        rationale=featured.rationale,
    )
    task_tuples.append(build_structure_task(featured_decision))
    
    # Other sections (skip secondary regions in COMBINED)
    for section in state.document_skeleton.sections:
        # Skip if this region is the secondary part of a COMBINED section
        if section.region in combined_secondary_regions:
            logger.info(f"   ⏭️ [{section.region}] Skipping - combined with another region")
            continue
        
        # For COMBINED sections, merge events from both clusters
        if section.treatment == "COMBINED" and section.combine_with:
            # combine_with is a cluster_id like "cluster_24"
            primary_cluster_id = section.source_cluster_id or section.region
            secondary_cluster_id = section.combine_with
            
            primary_events = cluster_events_map.get(primary_cluster_id, [])
            secondary_events = cluster_events_map.get(secondary_cluster_id, [])
            combined_events = primary_events + secondary_events
            
            # Store merged events for Writer to use (keyed by primary cluster)
            cluster_events_map[primary_cluster_id] = combined_events
            logger.debug(f"   🔗 [{section.region}] Combined {len(primary_events)} + {len(secondary_events)} events")
        
        task_tuples.append(build_structure_task(section))
    
    # Execute with orchestrator or fallback
    if orchestrator:
        state.section_blueprints = await orchestrator.run_agents_batch(task_tuples, "structure")
    else:
        from orchestrator import run_agents_parallel
        agent_results = await run_agents_parallel(task_tuples)
        state.section_blueprints = {name: r.result for name, r in agent_results.items()}


# =============================================================================
# PHASE 4: CONTENT PIPELINE (Writer Loop + Stylist Loop per section)
# =============================================================================


async def run_content_pipelines(
    state: PipelineState,
    cluster_events_map: dict[str, list[dict]],
    max_writer_attempts: int | None = None,
    max_stylist_attempts: int | None = None,
) -> None:
    """
    Run the integrated Content Pipeline for each section.
    
    Each section runs:
    1. Writer Loop (Writer + ContentCritic with retries)
    2. Stylist Loop (Stylist + StyleCritic with retries)
    
    This replaces the old Phase 4 + Phase 5 with an integrated approach.
    """
    from google import genai as genai_client
    from state import ContentCriticResult, StyleCriticResult
    
    config = get_config()
    max_writer_attempts = max_writer_attempts or config.max_retry_attempts
    max_stylist_attempts = max_stylist_attempts or config.max_retry_attempts
    
    # Build map of COMBINED sections (primary cluster_id -> secondary cluster_id)
    combined_map = {}
    if state.document_skeleton:
        for section in state.document_skeleton.sections:
            if section.treatment == "COMBINED" and section.combine_with:
                combined_map[section.source_cluster_id] = section.combine_with
    
    # Create shared client for all pipelines
    client = genai_client.Client(api_key=config.gemini_api_key)
    
    async def process_section(cluster_id: str) -> tuple[str, str, str, ContentCriticResult | None, StyleCriticResult | None]:
        """
        Run content pipeline for a single section.
        
        Returns: (cluster_id, writer_draft, styled_draft, content_result, style_result)
        """
        blueprint = state.section_blueprints.get(cluster_id)
        analyst_output = state.analyst_outputs.get(cluster_id)
        events = cluster_events_map.get(cluster_id, [])
        
        # Find the display region name for logging
        display_name = cluster_id
        if state.document_skeleton:
            if state.document_skeleton.featured.source_cluster_id == cluster_id:
                display_name = state.document_skeleton.featured.region
            else:
                for section in state.document_skeleton.sections:
                    if section.source_cluster_id == cluster_id:
                        display_name = section.region
                        break
        
        logger.info(f"   📦 [{display_name}] Starting content pipeline with {len(events)} events")
        
        # Handle COMBINED sections
        if cluster_id in combined_map:
            logger.debug(f"   🔗 [{display_name}] Combined section (events pre-merged)")
        
        if not blueprint or not analyst_output:
            state.add_warning(f"Missing data for {cluster_id} ({display_name})")
            return cluster_id, "", "", None, None
        
        # Extract sacred elements BEFORE running pipeline
        sacred = extract_sacred_elements(analyst_output, events)
        state.sacred_elements[cluster_id] = sacred
        
        # Determine if this is featured section
        is_featured = (
            state.document_skeleton and 
            state.document_skeleton.featured.source_cluster_id == cluster_id
        )
        
        # Get editorial angle for featured sections
        editorial_angle = None
        if is_featured and state.document_skeleton:
            editorial_angle = state.document_skeleton.featured.angle
        
        # Run the integrated content pipeline
        try:
            writer_draft, styled_draft, content_result, style_result = await asyncio.wait_for(
                run_content_pipeline(
                    analyst_output=analyst_output,
                    events=events,
                    blueprint=blueprint,
                    sacred_elements=sacred,
                    editorial_angle=editorial_angle,
                    max_writer_attempts=max_writer_attempts,
                    max_stylist_attempts=max_stylist_attempts,
                    client=client,
                ),
                timeout=1200  # 20 minutes for full pipeline
            )
            
            logger.info(
                f"   ✅ [{display_name}] Pipeline complete: "
                f"content={content_result.score}/100, style={style_result.score}/100"
            )
            
            return cluster_id, writer_draft, styled_draft, content_result, style_result
            
        except asyncio.TimeoutError:
            logger.error(f"   ⏰ [{display_name}] Content pipeline timed out after 1200s")
            state.add_error(f"Content pipeline timed out for {cluster_id}")
            return cluster_id, f"[Pipeline timed out for {display_name}]", "", None, None
            
        except Exception as e:
            logger.error(f"   ❌ [{display_name}] Content pipeline failed: {e}")
            state.add_error(f"Content pipeline failed for {cluster_id}: {e}")
            return cluster_id, f"[Pipeline failed for {display_name}]", "", None, None
    
    # Process all sections in parallel
    clusters_to_process = list(state.section_blueprints.keys())
    results = await asyncio.gather(*[process_section(cid) for cid in clusters_to_process])
    
    # Store results in state
    for cluster_id, writer_draft, styled_draft, content_result, style_result in results:
        if writer_draft:
            state.writer_drafts[cluster_id] = writer_draft
        if styled_draft:
            state.styled_drafts[cluster_id] = styled_draft
        if content_result and style_result:
            # Store combined result for backwards compatibility
            from state import CriticResult, CoVeResult
            state.critique_results[cluster_id] = CriticResult(
                passed=content_result.passed and style_result.passed,
                overall_score=(content_result.score + style_result.score) // 2,
                content_passed=content_result.passed,
                content_score=min(50, content_result.score // 2),
                content_issues=content_result.issues,
                style_passed=style_result.passed,
                style_score=min(50, style_result.score // 2),
                style_issues=style_result.issues,
                cove_result=content_result.cove_result,
                feedback_for_writer=content_result.feedback,
                feedback_for_stylist=style_result.feedback,
                content_result=content_result,
                style_result=style_result,
            )


# =============================================================================
# PHASE 6: ASSEMBLY
# =============================================================================


def assemble_document(
    state: PipelineState,
    calendar_events: list[dict],
    event_count: int,
    source_count: int,
) -> str:
    """Assemble final document using DocumentSkeleton."""
    
    if not state.document_skeleton:
        # Fallback to simple assembly
        from agents.schemas import SectionOutput
        sections = []
        for region, content in state.styled_drafts.items():
            sections.append(SectionOutput(
                region=region,
                content=content,
                analyst_data=state.analyst_outputs.get(region),
                critic_score=state.critique_results.get(region, {}).overall_score if state.critique_results.get(region) else 0,
            ))
        
        return assemble_document(
            featured_section=sections[0] if sections else None,
            regional_sections=sections[1:],
            cross_regional_connections=[],
            calendar_events=calendar_events,
            event_count=event_count,
            source_count=source_count,
        )
    
    skeleton = state.document_skeleton
    
    # Build mappings for content lookup
    # styled_drafts are now keyed by cluster_id (e.g. "cluster_24")
    cluster_to_headline: dict[str, str] = {}
    cluster_to_geo_region: dict[str, str] = {}
    if skeleton.featured.source_cluster_id:
        cluster_to_headline[skeleton.featured.source_cluster_id] = skeleton.featured.headline
        cluster_to_geo_region[skeleton.featured.source_cluster_id] = skeleton.featured.region
    for section in skeleton.sections:
        if section.source_cluster_id:
            cluster_to_headline[section.source_cluster_id] = section.headline
            cluster_to_geo_region[section.source_cluster_id] = section.region
    
    logger.debug(f"   🗺️ Assembly cluster_id→headline map: {cluster_to_headline}")
    
    def get_styled_content(cluster_id: str) -> str:
        """Get styled content by cluster_id (primary key)."""
        return state.styled_drafts.get(cluster_id, "")
    lines = []
    
    # Header
    lines.append("# 🌍 State of the World")
    lines.append("")
    lines.append(f"*Weekly Intelligence Briefing | {datetime.now(timezone.utc).strftime('%B %d, %Y')}*")
    lines.append("")
    
    # Executive Summary (expanded from narrative arc)
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    
    # Build executive summary from:
    # 1. Narrative arc (the through-line)
    # 2. Featured headline (the lead story)
    # 3. Top 2 section themes
    exec_summary_parts = []
    exec_summary_parts.append(skeleton.narrative_arc)
    
    # Add featured story context
    if skeleton.featured.headline:
        # Extract just the first sentence of the featured section as teaser
        featured_cluster_id = skeleton.featured.source_cluster_id
        featured_content = get_styled_content(featured_cluster_id) if featured_cluster_id else ""
        if featured_content:
            # Get first substantial sentence (skip headlines/formatting)
            import re
            sentences = re.split(r'[.!?]\s+', featured_content)
            teaser = next((s for s in sentences if len(s) > 50 and not s.startswith('#') and not s.startswith('<!--')), "")
            if teaser:
                exec_summary_parts.append(
                    f"The week's defining story: {skeleton.featured.headline} — {teaser[:150]}..."
                )
            else:
                exec_summary_parts.append(
                    f"The week's defining story: {skeleton.featured.headline}."
                )
        else:
            exec_summary_parts.append(
                f"The week's defining story: {skeleton.featured.headline}."
            )
    
    # Add key section themes (up to 2, excluding featured headline to avoid duplication)
    featured_headline = skeleton.featured.headline if skeleton.featured else ""
    section_themes = []
    for section in skeleton.sections:
        if len(section_themes) >= 2:
            break
        # Skip if same headline as featured (avoid duplication)
        if section.headline and section.headline != featured_headline:
            section_themes.append(section.headline)
    if section_themes:
        exec_summary_parts.append(f"Also watch: {', '.join(section_themes)}.")
    
    lines.append(" ".join(exec_summary_parts))
    lines.append("")
    
    # Handle organization based on organizing_principle
    if skeleton.organizing_principle == "THEMATIC":
        # Hub and Spoke Model
        # THE HUB: The abstract mechanism (NOT a story or actor)
        # hub_mechanism is the force/pattern, featured is Spoke 1
        
        lines.append("---")
        lines.append("")
        
        # Use hub_mechanism if defined, otherwise fall back (backward compat)
        hub_mech = getattr(skeleton, 'hub_mechanism', None) or getattr(skeleton, 'hub_theme', None)
        
        if hub_mech:
            lines.append(f"## 🎯 The Hub: {hub_mech}")
            lines.append("")
            lines.append(f"*The central mechanism driving this week's developments*")
            lines.append("")
            if skeleton.hub_explanation:
                lines.append(skeleton.hub_explanation)
                lines.append("")
            
            lines.append("---")
            lines.append("")
            
            # Now add the featured region as SPOKE 1 (most important manifestation)
            featured_cid = skeleton.featured.source_cluster_id
            featured_content = get_styled_content(featured_cid) if featured_cid else ""
            if featured_content:
                lines.append(f"### Spoke 1 (Featured): {skeleton.featured.headline}")
                lines.append("")
                lines.append(featured_content)
                lines.append("")
        else:
            # Fallback for old skeletons without hub (backward compat)
            featured_cid = skeleton.featured.source_cluster_id
            featured_content = get_styled_content(featured_cid) if featured_cid else ""
            if featured_content:
                lines.append(f"## 🎯 The Hub: {skeleton.featured.headline}")
                lines.append("")
                lines.append(f"*The central mechanism driving this week's developments*")
                lines.append("")
                lines.append(featured_content)
                lines.append("")
        
        # THE SPOKES: Regional manifestations organized by section groups
        lines.append("---")
        lines.append("")
        lines.append("## 🔀 The Spokes: Regional Manifestations")
        lines.append("")
        lines.append("*How the hub mechanism manifests across regions*")
        lines.append("")
        
        spoke_counter = 2  # Spoke 1 is featured
        
        for group in skeleton.section_groups:
            # section_groups.sections contains cluster_ids (e.g. "cluster_20")
            # Skip featured cluster_id in spokes (already shown as Spoke 1)
            featured_cid = skeleton.featured.source_cluster_id
            non_featured_cluster_ids = [cid for cid in group.sections if cid != featured_cid]
            if not non_featured_cluster_ids:
                continue
            
            lines.append(f"### {group.title}")
            if group.group_rationale:
                lines.append(f"*{group.group_rationale}*")
            lines.append("")
            
            prev_cluster_id = None
            for cluster_id in non_featured_cluster_ids:
                # Add transition if available
                if prev_cluster_id:
                    transition_key = f"{prev_cluster_id} → {cluster_id}"
                    transition = skeleton.transitions.get(transition_key, "")
                    if transition and transition != "HARD_BREAK":
                        lines.append(f"*{transition}*")
                        lines.append("")
                
                # Find section headline from cluster_to_headline map
                section_headline = cluster_to_headline.get(cluster_id, cluster_id)
                
                # Get content by cluster_id (our primary key)
                content = get_styled_content(cluster_id)
                if content:
                    lines.append(f"#### Spoke {spoke_counter}: {section_headline}")
                    spoke_counter += 1
                    lines.append("")
                    lines.append(content)
                    lines.append("")
                else:
                    logger.warning(f"   ⚠️ No content found for spoke: {cluster_id} ({section_headline})")
                
                prev_cluster_id = cluster_id
    
    else:
        # REGIONAL or HYBRID organization (standard)
        # Featured
        featured_cid = skeleton.featured.source_cluster_id
        featured_content = get_styled_content(featured_cid) if featured_cid else ""
        if featured_content:
            lines.append("---")
            lines.append("")
            lines.append(f"## ⭐ Featured: {skeleton.featured.headline}")
            lines.append("")
            lines.append(featured_content)
            lines.append("")
        
        # Section groups
        for group in skeleton.section_groups:
            lines.append("---")
            lines.append("")
            lines.append(f"## {group.title}")
            if group.group_rationale:
                lines.append(f"*{group.group_rationale}*")
            lines.append("")
            
            featured_cid = skeleton.featured.source_cluster_id
            prev_cluster_id = None
            for cluster_id in group.sections:
                if cluster_id == featured_cid:
                    continue  # Already handled featured
                
                # Add transition if available
                if prev_cluster_id:
                    transition_key = f"{prev_cluster_id} → {cluster_id}"
                    transition = skeleton.transitions.get(transition_key, "")
                    if transition and transition != "HARD_BREAK":
                        lines.append(f"*{transition}*")
                        lines.append("")
                
                # Get headline from our map
                section_headline = cluster_to_headline.get(cluster_id, cluster_id)
                
                # Add section content (keyed by cluster_id)
                content = get_styled_content(cluster_id)
                if content:
                    lines.append(f"### {section_headline}")
                    lines.append("")
                    lines.append(content)
                    lines.append("")
                else:
                    logger.warning(f"   ⚠️ No content found for section: {cluster_id} ({section_headline})")
                
                prev_cluster_id = cluster_id
    
    # Quick Hits
    if skeleton.quick_hits:
        lines.append("---")
        lines.append("")
        lines.append("## ⚡ Quick Hits")
        lines.append("")
        for qh in skeleton.quick_hits:
            lines.append(f"- **{qh.region}:** {qh.content}")
        lines.append("")
    
    # Methodology
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    
    # Use total_event_count from state (set in Phase 1)
    event_count = getattr(state, 'total_event_count', 0)
    sections_analyzed = len(skeleton.sections) + 1  # +1 for featured
    sections_killed = len(skeleton.killed)
    
    lines.append(
        f"This briefing analyzed **{event_count} events** from realpolitik.world using a 10-phase "
        f"multi-agent editorial pipeline with kill authority."
    )
    lines.append("")
    lines.append(
        f"**Editorial decisions:** {sections_analyzed} sections published, {sections_killed} clusters killed via PIC Matrix scoring."
    )
    lines.append("")
    lines.append(
        "**Analysis frameworks:** Constraints-of-Thought, Futures Wheel, Analysis of Competing Hypotheses, "
        "Chain of Verification (CoVe) for fact-checking."
    )
    lines.append("")
    lines.append(
        "**Probability language:** Sherman Kent estimative probability scale "
        "(ALMOST CERTAIN 93-99%, HIGHLY LIKELY 80-92%, LIKELY 60-79%, etc.)."
    )
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%B %d, %Y')}*")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN PIPELINE
# =============================================================================


async def run_pipeline(
    mode: str = "test",
    dry_run: bool = True,
    max_phase: int = 7,
    skip_images: bool = True,
) -> int:
    """
    Run the full Briefing pipeline.
    
    Phases:
        1. Aggregate: Collect events
        2. Cluster: Semantic clustering
        3. Analyze: Run Analyst agents (parallel)
        4. Architect: Editorial decisions
        5. Structure: Beat sheet planning
        6. Draft: Writer + Stylist
        7. Critique: CoVe + feedback routing
        8. Assemble: Final document
        9. Images: Generate visuals
        10. Store: Save/publish
    """
    # Initialize config
    reset_config()
    reset_cache()
    clear_checkpoints()
    
    config = get_config(mode=mode)
    
    print_banner(mode)
    
    start_time = time.time()
    
    # Initialize state and orchestrator
    run_id = f"briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = Path(__file__).parent / "outputs" / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging to capture all output
    log_file = setup_file_logging(run_output_dir)
    
    try:
        return await _run_pipeline_inner(mode, dry_run, max_phase, run_output_dir, start_time)
    finally:
        # Always close file logging, even on early return or exception
        close_file_logging()


async def _run_pipeline_inner(
    mode: str,
    dry_run: bool,
    max_phase: int,
    run_output_dir: Path,
    start_time: float,
) -> int:
    """Inner pipeline function wrapped by run_pipeline for proper cleanup."""
    
    run_id = run_output_dir.name
    
    # Reset search results storage for this run (for CoVe transparency)
    reset_search_results()
    
    state = PipelineState(
        run_id=run_id,
        started_at=datetime.now(timezone.utc),
    )
    orchestrator = create_orchestrator(state, output_dir=run_output_dir)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: AGGREGATION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📥 PHASE 1: Aggregation")
    logger.info("=" * 60)
    
    week_data = aggregate_week()
    event_count = len(week_data.events)
    source_count = len(set(e.get("source", "") for e in week_data.events))
    
    # Store event counts in state for later use (e.g., methodology section)
    state.total_event_count = event_count
    state.total_source_count = source_count
    
    logger.info(f"   ✅ Collected {event_count} events from {source_count} sources")
    logger.info(f"   ✅ Regions: {list(week_data.by_region.keys())}")
    
    orchestrator.save_checkpoint("phase_1_aggregate")
    
    if max_phase == 1:
        logger.info("\n✅ Stopping at phase 1")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: CLUSTERING
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("🔬 PHASE 2: Semantic Clustering")
    logger.info("=" * 60)
    
    cluster_analysis = cluster_week(week_data.events, week_data.by_region)
    storyline_count = sum(len(r.storylines) for r in cluster_analysis.regions.values())
    
    # Build cluster_events_map - THE canonical source of events for each cluster
    # This replaces the broken regions_data lookups throughout the pipeline
    cluster_events_map: dict[str, list[dict]] = {}
    for cluster in cluster_analysis.thematic_clusters:
        cluster_key = f"cluster_{cluster.cluster_id}"
        cluster_events_map[cluster_key] = cluster.events
        logger.debug(f"   📦 {cluster_key}: {len(cluster.events)} events")
    
    logger.info(f"   ✅ Created {storyline_count} storylines")
    logger.info(f"   ✅ Cross-regional connections: {len(cluster_analysis.cross_regional_connections)}")
    logger.info(f"   📦 Cluster events map: {len(cluster_events_map)} clusters")
    
    orchestrator.save_checkpoint("phase_2_cluster")
    
    if max_phase == 2:
        logger.info("\n✅ Stopping at phase 2")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3A: EDITOR AGENT (Editorial Research & Decisions)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✏️  PHASE 3A: Editor Agent (Research & Decisions)")
    logger.info("=" * 60)
    
    # Get thematic clustering data
    thematic_clusters = [c.to_dict() for c in cluster_analysis.thematic_clusters]
    hub_candidates = [c.to_dict() for c in cluster_analysis.hub_candidates]
    recommended_org = cluster_analysis.recommended_organization
    
    cross_connections = [
        conn.explanation if hasattr(conn, 'explanation') else str(conn)
        for conn in cluster_analysis.cross_regional_connections
    ]
    
    logger.info(f"   🎯 Thematic clusters: {len(thematic_clusters)}")
    logger.info(f"   🎯 Hub candidates: {len(hub_candidates)}")
    logger.info(f"   🎯 Recommended organization: {recommended_org}")
    
    # Run Editor agent
    await run_editor(
        state,
        thematic_clusters=thematic_clusters,
        hub_candidates=hub_candidates,
        recommended_organization=recommended_org,
        cross_regional_connections=cross_connections,
        calendar_events=[],  # Calendar disabled
        orchestrator=orchestrator,
    )
    
    orchestrator.save_checkpoint("phase_3a_editor")
    
    if max_phase == 3:
        logger.info("\n✅ Stopping at phase 3")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3B: ARCHITECT AGENT (Structure into JSON)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📐 PHASE 3B: Architect Agent (Structure Decisions)")
    logger.info("=" * 60)
    
    # Run Architect agent (structures Editor decisions into JSON)
    await run_architect(state, orchestrator=orchestrator)
    
    if state.document_skeleton:
        logger.info(f"   ✅ Featured: {state.document_skeleton.featured.region}")
        logger.info(f"   ✅ Publishing: {len(state.document_skeleton.sections)} sections")
        logger.info(f"   🚫 Killed: {len(state.document_skeleton.killed)} clusters")
        logger.info(f"   📝 Narrative: {state.document_skeleton.narrative_arc[:60]}...")
    
    orchestrator.save_checkpoint("phase_3b_architect")
    
    if max_phase == 3:
        logger.info("\n✅ Stopping at phase 3 (after Architect)")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: ANALYST AGENTS (only for PUBLISHED clusters)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("🧠 PHASE 4: Analyst Agents (Published Clusters Only)")
    logger.info("=" * 60)
    
    # Get only the clusters that the Architect decided to publish
    # Use source_cluster_id (cluster_XX) for reliable matching
    published_cluster_ids = set()
    if state.document_skeleton:
        # Featured region
        if state.document_skeleton.featured.source_cluster_id:
            published_cluster_ids.add(state.document_skeleton.featured.source_cluster_id)
        # All sections
        for section in state.document_skeleton.sections:
            if section.source_cluster_id:
                published_cluster_ids.add(section.source_cluster_id)
    
    # Filter thematic clusters to only published ones
    # Match by cluster_id (canonical key)
    published_clusters = [
        c for c in cluster_analysis.thematic_clusters
        if f"cluster_{c.cluster_id}" in published_cluster_ids
    ]
    
    logger.info(f"   📊 Analyzing {len(published_clusters)} published clusters (skipping {len(cluster_analysis.thematic_clusters) - len(published_clusters)} killed)")
    
    if published_clusters:
        state.analyst_outputs = await run_analysts_on_themes(state, published_clusters, orchestrator)
        logger.info(f"   ✅ Analyzed {len(state.analyst_outputs)} clusters")
    else:
        logger.warning("   ⚠️ No published clusters to analyze")
        state.analyst_outputs = {}
    
    orchestrator.save_checkpoint("phase_4_analysts")
    
    if max_phase == 4:
        logger.info("\n✅ Stopping at phase 4")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: STRUCTURE AGENTS
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📝 PHASE 5: Structure Agents (Parallel)")
    logger.info("=" * 60)
    
    await run_structure_parallel(state, cluster_events_map, orchestrator)
    
    logger.info(f"   ✅ Created {len(state.section_blueprints)} blueprints")
    
    orchestrator.save_checkpoint("phase_5_structure")
    
    if max_phase == 5:
        logger.info("\n✅ Stopping at phase 5")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 6: CONTENT PIPELINE (Writer Loop + Stylist Loop with Critics)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✍️ PHASE 6: Content Pipeline (Writer+Critic → Stylist+Critic)")
    logger.info("=" * 60)
    
    await run_content_pipelines(state, cluster_events_map)
    
    logger.info(f"   ✅ Writer drafts: {len(state.writer_drafts)}")
    logger.info(f"   ✅ Styled drafts: {len(state.styled_drafts)}")
    
    orchestrator.save_checkpoint("phase_6_content_pipeline")
    
    if max_phase == 6:
        logger.info("\n✅ Stopping at phase 6")
        return 0
    
    # Save search results log for transparency
    save_search_results_log(str(run_output_dir))
    
    # Log verification corpus info
    all_search_results = get_all_search_results()
    total_events_in_map = sum(len(events) for events in cluster_events_map.values())
    logger.info(f"   📚 Verification corpus: {total_events_in_map} cluster events, {len(all_search_results)} web searches")
    
    # Summary
    passed = sum(1 for r in state.critique_results.values() if r.passed)
    total = len(state.critique_results)
    avg_score = sum(r.overall_score for r in state.critique_results.values()) / total if total else 0
    
    logger.info(f"   ✅ Passed: {passed}/{total} sections")
    logger.info(f"   📊 Average score: {avg_score:.1f}/100")
    
    # Check for hallucinations
    hallucinations = sum(
        len(r.cove_result.hallucination_flags) 
        for r in state.critique_results.values()
    )
    if hallucinations:
        logger.warning(f"   ⚠️ Hallucination flags: {hallucinations}")
    
    if max_phase == 7:
        logger.info("\n✅ Stopping at phase 7 (no separate phase 7 anymore)")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7: ASSEMBLY
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📄 PHASE 7: Document Assembly")
    logger.info("=" * 60)
    
    document = assemble_document(
        state,
        [],  # Calendar disabled
        event_count,
        source_count,
    )
    
    state.final_output = document
    
    word_count = len(document.split())
    logger.info(f"   ✅ Assembled document: {word_count} words")
    
    orchestrator.save_checkpoint("phase_7_assembly")
    
    if max_phase == 8:
        logger.info("\n✅ Stopping at phase 8 (assembly complete)")
        print("\n" + "=" * 60)
        print(document)
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 8: IMAGES (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if skip_images:
        logger.info("\n" + "=" * 60)
        logger.info("🎨 PHASE 8: Image Generation (SKIPPED)")
        logger.info("=" * 60)
        logger.info("   ⏭️ Skipping image generation (use --generate-images to enable)")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("🎨 PHASE 8: Image Generation")
        logger.info("=" * 60)
        
        # Create images subdirectory
        images_dir = run_output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Image enhancement
        from agents.image_prompts import enhance_all_image_prompts
        
        featured_region = state.document_skeleton.featured.region if state.document_skeleton else None
        document, enhanced_prompts = await enhance_all_image_prompts(
            document,
            region_name=featured_region,
        )
        
        if enhanced_prompts:
            logger.info(f"   📷 Enhanced {len(enhanced_prompts)} image prompts")
        
        # Generate images
        if dry_run:
            generated = generate_briefing_images(document, str(images_dir))
            logger.info(f"   ✅ Generated {len(generated)} images to {images_dir}")
        else:
            # Production: upload to R2
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                generated = generate_briefing_images(document, temp_dir)
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                
                image_urls = []
                for location, filepath in generated:
                    with open(filepath, "rb") as f:
                        image_data = f.read()
                    
                    filename = Path(filepath).name
                    r2_key = f"briefing/{date_str}/images/{filename}"
                    
                    try:
                        url = upload_image_to_r2(image_data, r2_key)
                        image_urls.append((location, url))
                        logger.info(f"   ✅ Uploaded: {url}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Upload failed: {e}")
                
                if image_urls:
                    document = insert_images_into_markdown(document, image_urls, base_url="")
    
    orchestrator.save_checkpoint("phase_8_images")
    
    if max_phase == 9:
        logger.info("\n✅ Stopping at phase 9 (images complete)")
        return 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 9: STORAGE
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("💾 PHASE 9: Storage")
    logger.info("=" * 60)
    
    if dry_run:
        # Save locally
        output_path = run_output_dir / "briefing.md"
        output_path.write_text(document)
        logger.info(f"   ✅ Saved to: {output_path}")
        
        # Save metadata
        metadata = {
            "run_id": state.run_id,
            "version": "3.1",
            "event_count": event_count,
            "source_count": source_count,
            "regions_analyzed": list(state.analyst_outputs.keys()),
            "regions_published": [s.region for s in state.document_skeleton.sections] if state.document_skeleton else [],
            "regions_killed": [k.region for k in state.document_skeleton.killed] if state.document_skeleton else [],
            "narrative_arc": state.document_skeleton.narrative_arc if state.document_skeleton else "",
            "critic_scores": {k: v.overall_score for k, v in state.critique_results.items()},
            "errors": state.errors,
            "warnings": state.warnings,
            "elapsed_seconds": time.time() - start_time,
        }
        
        metadata_path = run_output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info(f"   ✅ Metadata saved to: {metadata_path}")
    else:
        # Production: save to R2 and notify
        try:
            save_briefing(document, metadata={})
            logger.info("   ✅ Published to production")
        except Exception as e:
            logger.error(f"   ❌ Publication failed: {e}")
            return 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMPLETE
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    metrics_summary = orchestrator.complete()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   ⏱️ Total time: {elapsed:.1f}s")
    logger.info(f"   📊 Regions: {len(state.analyst_outputs)} analyzed, "
                f"{len(state.styled_drafts)} published")
    logger.info(f"   📝 Document: {len(document.split())} words")
    
    # Log phase metrics
    for phase_name, phase_data in metrics_summary.get("phases", {}).items():
        success_rate = phase_data.get("success_rate", 0) * 100
        logger.info(f"   📈 {phase_name}: {phase_data.get('duration_seconds', 0):.1f}s, {success_rate:.0f}% success")
    
    if state.errors:
        logger.warning(f"   ⚠️ Errors: {len(state.errors)}")
    
    return 0


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="The Briefing: Multi-Agent Intelligence Analysis Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "production"],
        default="test",
        help="Pipeline mode (test uses cheaper models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Save locally instead of publishing",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Publish to production",
    )
    parser.add_argument(
        "--max-phase",
        type=int,
        default=10,
        help="Stop after this phase (for debugging)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        default=True,
        help="Skip image generation phase (default: True)",
    )
    parser.add_argument(
        "--generate-images",
        action="store_false",
        dest="skip_images",
        help="Enable image generation",
    )
    
    args = parser.parse_args()
    
    return asyncio.run(run_pipeline(
        mode=args.mode,
        dry_run=args.dry_run,
        max_phase=args.max_phase,
        skip_images=args.skip_images,
    ))


if __name__ == "__main__":
    sys.exit(main())
