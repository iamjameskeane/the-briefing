"""
The Briefing - Pipeline Orchestrator

Handles:
- State management and checkpointing
- Agent routing with timeouts
- Parallel execution
- Retry logic with feedback routing
- Graceful degradation

Separates orchestration concerns from phase definitions in run.py.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from config import get_config
from state import PipelineState, SacredElements
from utils import logger

if TYPE_CHECKING:
    from agents.schemas import AnalystOutput
    from state import SectionBlueprint, CriticResult


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar("T")


# =============================================================================
# CHECKPOINTING
# =============================================================================


@dataclass
class CheckpointManager:
    """
    Manages pipeline checkpoints for debugging and recovery.
    
    Saves run-specific checkpoints:
    1. Minimal flags in outputs/{run_id}/.checkpoints/ (for recovery)
    2. Full agent outputs in outputs/{run_id}/agent_outputs/ (for debugging)
    
    This ensures each run has isolated checkpoints and can be resumed independently.
    """
    
    checkpoint_dir: Path | None = None  # Set to outputs/{run_id}/.checkpoints/
    output_dir: Path | None = None  # Set to outputs/{run_id}/
    
    def save(self, state: PipelineState, phase: str) -> None:
        """
        Save pipeline state after a phase completes.
        
        Saves minimal checkpoint to outputs/{run_id}/.checkpoints/ for recovery,
        and detailed agent outputs to outputs/{run_id}/agent_outputs/ for debugging.
        """
        if not self.checkpoint_dir:
            logger.warning("⚠️  Checkpoint directory not set, skipping checkpoint save")
            return
        
        # Minimal checkpoint for recovery
        self.checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = self.checkpoint_dir / f"{phase}.json"
        
        state.add_checkpoint(phase)
        
        # Serialize key state flags
        checkpoint_data = {
            "run_id": state.run_id,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "regions_analyzed": list(state.analyst_outputs.keys()),
            "has_skeleton": state.document_skeleton is not None,
            "blueprints_created": list(state.section_blueprints.keys()),
            "drafts_created": list(state.writer_drafts.keys()),
            "styled_drafts": list(state.styled_drafts.keys()),
            "critique_results": {
                k: v.overall_score for k, v in state.critique_results.items()
            },
            "errors": state.errors,
            "warnings": state.warnings,
        }
        
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2, default=str))
        
        # Detailed agent outputs for debugging (if output_dir is set)
        if self.output_dir:
            self._save_detailed_checkpoint(state, phase)
        
        logger.info(f"   💾 Checkpoint: {phase}")
    
    def _save_detailed_checkpoint(self, state: PipelineState, phase: str) -> None:
        """Save detailed agent outputs for debugging."""
        agent_output_dir = self.output_dir / "agent_outputs"
        agent_output_dir.mkdir(exist_ok=True)
        
        # Phase-specific output saving
        if phase == "phase_3a_editor" and state.editor_decisions:
            # Save Editor's editorial brief
            output_file = agent_output_dir / "01_editor_brief.txt"
            output_file.write_text(state.editor_decisions.editorial_brief)
            
            # Save Editor's conversation summary
            convo_file = agent_output_dir / "01_editor_conversation.json"
            convo_data = {
                "message_count": len(state.editor_decisions.conversation_history),
                "brief_length": len(state.editor_decisions.editorial_brief),
            }
            convo_file.write_text(json.dumps(convo_data, indent=2))
        
        elif phase == "phase_3b_architect":
            # Save Architect's reasoning (step 1 output)
            if state.architect_reasoning:
                reasoning_file = agent_output_dir / "02a_architect_reasoning.txt"
                reasoning_file.write_text(state.architect_reasoning)
            
            # Save Architect's DocumentSkeleton (step 2 output)
            if state.document_skeleton:
                output_file = agent_output_dir / "02b_document_skeleton.json"
                skeleton_data = state.document_skeleton.model_dump()
                output_file.write_text(json.dumps(skeleton_data, indent=2, default=str))
        
        elif phase == "phase_4_analysts" and state.analyst_outputs:
            # Save all Analyst outputs
            for cluster_id, analyst_output in state.analyst_outputs.items():
                safe_id = cluster_id.replace("/", "_").replace(" ", "_")
                output_file = agent_output_dir / f"03_analyst_{safe_id}.json"
                if hasattr(analyst_output, 'model_dump'):
                    data = analyst_output.model_dump()
                elif hasattr(analyst_output, 'to_dict'):
                    data = analyst_output.to_dict()
                else:
                    data = vars(analyst_output)
                output_file.write_text(json.dumps(data, indent=2, default=str))
        
        elif phase == "phase_5_structure" and state.section_blueprints:
            # Save all Structure blueprints
            for region, blueprint in state.section_blueprints.items():
                safe_id = region.replace("/", "_").replace(" ", "_")
                output_file = agent_output_dir / f"04_blueprint_{safe_id}.json"
                if hasattr(blueprint, 'model_dump'):
                    data = blueprint.model_dump()
                elif hasattr(blueprint, 'to_dict'):
                    data = blueprint.to_dict()
                else:
                    data = vars(blueprint)
                output_file.write_text(json.dumps(data, indent=2, default=str))
        
        elif phase == "phase_6_drafting" and state.writer_drafts:
            # Save all Writer drafts
            for region, draft in state.writer_drafts.items():
                safe_id = region.replace("/", "_").replace(" ", "_")
                output_file = agent_output_dir / f"05_draft_{safe_id}.txt"
                output_file.write_text(draft)
            
            # Save styled drafts if available
            if state.styled_drafts:
                for region, styled in state.styled_drafts.items():
                    safe_id = region.replace("/", "_").replace(" ", "_")
                    output_file = agent_output_dir / f"06_styled_{safe_id}.txt"
                    output_file.write_text(styled)
        
        elif phase == "phase_7_critique" and state.critique_results:
            # Save all Critic results
            for region, critique in state.critique_results.items():
                safe_id = region.replace("/", "_").replace(" ", "_")
                output_file = agent_output_dir / f"07_critique_{safe_id}.json"
                if hasattr(critique, 'model_dump'):
                    data = critique.model_dump()
                elif hasattr(critique, 'to_dict'):
                    data = critique.to_dict()
                else:
                    data = vars(critique)
                output_file.write_text(json.dumps(data, indent=2, default=str))
    
    def clear(self) -> None:
        """Clear checkpoints for the current run."""
        if not self.checkpoint_dir:
            return
        
        if self.checkpoint_dir.exists():
            for f in self.checkpoint_dir.glob("*.json"):
                f.unlink()
            logger.info("   🧹 Cleared checkpoints for current run")
    
    def load(self, phase: str) -> dict | None:
        """Load a checkpoint if it exists."""
        if not self.checkpoint_dir:
            return None
        
        checkpoint_file = self.checkpoint_dir / f"{phase}.json"
        if checkpoint_file.exists():
            return json.loads(checkpoint_file.read_text())
        return None
    
    def list_checkpoints(self) -> list[str]:
        """List all saved checkpoints for the current run."""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return []
        return sorted(f.stem for f in self.checkpoint_dir.glob("*.json"))


# =============================================================================
# AGENT ROUTING
# =============================================================================


@dataclass
class AgentResult:
    """Result from an agent call with metadata."""
    
    success: bool
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    timed_out: bool = False


async def run_with_timeout(
    coro: Awaitable[T],
    timeout_seconds: int | None = None,
    fallback: T | None = None,
    name: str = "agent",
) -> AgentResult:
    """
    Run an async agent call with timeout and error handling.
    
    Args:
        coro: The async coroutine to run
        timeout_seconds: Max seconds to wait (None = use config default)
        fallback: Value to return on failure
        name: Name for logging
        
    Returns:
        AgentResult with success/failure info
    """
    timeout = timeout_seconds or 600  # 10 minutes default (generous for thinking + tools)
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        duration = asyncio.get_event_loop().time() - start_time
        return AgentResult(success=True, result=result, duration_seconds=duration)
        
    except asyncio.TimeoutError:
        duration = asyncio.get_event_loop().time() - start_time
        logger.error(f"   ⏰ [{name}] Timed out after {timeout}s")
        return AgentResult(
            success=False,
            result=fallback,
            error=f"Timed out after {timeout}s",
            duration_seconds=duration,
            timed_out=True,
        )
        
    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        logger.error(f"   ❌ [{name}] Failed: {e}")
        return AgentResult(
            success=False,
            result=fallback,
            error=str(e),
            duration_seconds=duration,
        )


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================


async def run_agents_parallel(
    tasks: list[tuple[str, Awaitable[T], T]],
    max_concurrent: int | None = None,
) -> dict[str, AgentResult]:
    """
    Run multiple agent tasks in parallel with concurrency limits.
    
    Args:
        tasks: List of (name, coroutine, fallback_value) tuples
        max_concurrent: Max concurrent tasks (None = config default)
        
    Returns:
        Dict mapping task name to AgentResult
    """
    config = get_config()
    max_concurrent = max_concurrent or config.max_concurrent_analysts
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, AgentResult] = {}
    
    async def run_one(name: str, coro: Awaitable[T], fallback: T) -> None:
        async with semaphore:
            result = await run_with_timeout(coro, name=name, fallback=fallback)
            results[name] = result
    
    await asyncio.gather(*[
        run_one(name, coro, fallback) 
        for name, coro, fallback in tasks
    ])
    
    return results


# =============================================================================
# RETRY ROUTING
# =============================================================================


@dataclass
class RetryDecision:
    """Decision on how to handle a failed critic check."""
    
    should_retry: bool
    route_to: str  # "writer" | "stylist" | "none"
    feedback: str | None = None
    max_retries_exceeded: bool = False


def route_critic_feedback(
    result: "CriticResult",
    writer_retries: int,
    stylist_retries: int,
    max_writer_retries: int = 2,
    max_stylist_retries: int = 2,
) -> RetryDecision:
    """
    Determine where to route critic feedback.
    
    Content issues → Writer retry
    Style issues → Stylist retry
    Both issues → Writer first
    """
    if result.passed:
        return RetryDecision(should_retry=False, route_to="none")
    
    # Content failed - route to Writer
    if not result.content_passed and writer_retries < max_writer_retries:
        return RetryDecision(
            should_retry=True,
            route_to="writer",
            feedback=result.feedback_for_writer,
        )
    
    # Style failed - route to Stylist
    if not result.style_passed and stylist_retries < max_stylist_retries:
        return RetryDecision(
            should_retry=True,
            route_to="stylist",
            feedback=result.feedback_for_stylist,
        )
    
    # Max retries exceeded
    return RetryDecision(
        should_retry=False,
        route_to="none",
        max_retries_exceeded=True,
    )


# =============================================================================
# FALLBACK HELPERS
# =============================================================================


def create_minimal_sacred_elements() -> SacredElements:
    """Create empty SacredElements for fallback scenarios."""
    return SacredElements(
        proper_nouns=[],
        statistics=[],
        dates=[],
        quotes=[],
        event_ids=[],
    )


# =============================================================================
# PIPELINE METRICS
# =============================================================================


@dataclass
class PhaseMetrics:
    """Metrics for a pipeline phase."""
    
    phase_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def complete(self) -> None:
        self.end_time = datetime.now()
    
    def record_result(self, result: AgentResult) -> None:
        if result.success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if result.timed_out:
                self.timeout_count += 1


@dataclass
class PipelineMetrics:
    """Aggregate metrics for the entire pipeline run."""
    
    run_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    phases: dict[str, PhaseMetrics] = field(default_factory=dict)
    
    def start_phase(self, phase_name: str) -> PhaseMetrics:
        metrics = PhaseMetrics(phase_name=phase_name)
        self.phases[phase_name] = metrics
        return metrics
    
    def complete_phase(self, phase_name: str) -> None:
        if phase_name in self.phases:
            self.phases[phase_name].complete()
    
    def complete(self) -> None:
        self.end_time = datetime.now()
    
    @property
    def total_duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "total_duration_seconds": self.total_duration_seconds,
            "phases": {
                name: {
                    "duration_seconds": m.duration_seconds,
                    "success_rate": m.success_rate,
                    "timeouts": m.timeout_count,
                }
                for name, m in self.phases.items()
            },
        }


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================


class PipelineOrchestrator:
    """
    Main orchestrator for The Briefing pipeline.
    
    Coordinates:
    - State management via checkpoints
    - Agent execution with timeouts
    - Parallel processing with concurrency limits
    - Retry routing for critic feedback
    - Metrics collection
    """
    
    def __init__(self, state: PipelineState, output_dir: Path | None = None):
        self.state = state
        
        # Set up run-specific checkpoint directory
        checkpoint_dir = output_dir / ".checkpoints" if output_dir else None
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir
        )
        
        self.metrics = PipelineMetrics(run_id=state.run_id)
        self.config = get_config()
        self.output_dir = output_dir
    
    def save_checkpoint(self, phase: str) -> None:
        """Save a checkpoint for the current phase."""
        self.checkpoint_manager.save(self.state, phase)
    
    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        self.checkpoint_manager.clear()
    
    async def run_agent(
        self,
        coro: Awaitable[T],
        fallback: T,
        name: str,
        timeout: int | None = None,
    ) -> T:
        """
        Run a single agent with timeout and fallback.
        
        Returns the result or fallback value.
        """
        result = await run_with_timeout(
            coro,
            timeout_seconds=timeout or 600,  # 10 minutes default
            fallback=fallback,
            name=name,
        )
        
        if not result.success:
            if result.error:
                self.state.add_error(f"{name}: {result.error}")
        
        return result.result
    
    async def run_agents_batch(
        self,
        tasks: list[tuple[str, Awaitable[T], T]],
        phase_name: str,
    ) -> dict[str, T]:
        """
        Run multiple agents in parallel with metrics tracking.
        
        Args:
            tasks: List of (name, coroutine, fallback) tuples
            phase_name: Name for metrics tracking
            
        Returns:
            Dict mapping name to result (or fallback)
        """
        phase_metrics = self.metrics.start_phase(phase_name)
        
        results = await run_agents_parallel(tasks)
        
        for name, agent_result in results.items():
            phase_metrics.record_result(agent_result)
            if not agent_result.success and agent_result.error:
                self.state.add_error(f"{name}: {agent_result.error}")
        
        phase_metrics.complete()
        
        # Return just the values (result or fallback)
        return {
            name: r.result for name, r in results.items()
        }
    
    def get_retry_decision(
        self,
        result: "CriticResult",
        writer_retries: int,
        stylist_retries: int,
    ) -> RetryDecision:
        """Get routing decision for critic feedback."""
        return route_critic_feedback(
            result,
            writer_retries,
            stylist_retries,
            max_writer_retries=self.config.max_retry_attempts,
            max_stylist_retries=self.config.max_retry_attempts,
        )
    
    def complete(self) -> dict:
        """Mark pipeline complete and return metrics summary."""
        self.metrics.complete()
        return self.metrics.summary()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_orchestrator(state: PipelineState, output_dir: Path | None = None) -> PipelineOrchestrator:
    """Create a new orchestrator for a pipeline run."""
    return PipelineOrchestrator(state, output_dir=output_dir)
