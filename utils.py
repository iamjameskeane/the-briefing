"""
The Briefing Utilities

Shared utilities for The Briefing pipeline including retry logic, logging, and caching.
"""

import functools
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("briefing")

# Track file handler for cleanup
_file_handler: logging.FileHandler | None = None


def log_model_config(agent_name: str, model: str, config: "types.GenerateContentConfig") -> None:
    """
    Log the model configuration parameters for debugging.
    
    Args:
        agent_name: Name of the agent (e.g., "Editor", "Architect")
        model: Model name being used
        config: GenerateContentConfig object
    """
    params = []
    
    if hasattr(config, 'temperature') and config.temperature is not None:
        params.append(f"temp={config.temperature}")
    
    if hasattr(config, 'max_output_tokens') and config.max_output_tokens is not None:
        params.append(f"max_tokens={config.max_output_tokens}")
    
    if hasattr(config, 'thinking_config') and config.thinking_config is not None:
        if hasattr(config.thinking_config, 'thinking_level') and config.thinking_config.thinking_level:
            params.append(f"thinking={config.thinking_config.thinking_level}")
        elif hasattr(config.thinking_config, 'thinking_budget'):
            params.append(f"thinking_budget={config.thinking_config.thinking_budget}")
    
    if hasattr(config, 'response_mime_type') and config.response_mime_type:
        params.append(f"mime={config.response_mime_type}")
    
    param_str = ", ".join(params) if params else "defaults"
    logger.info(f"   🤖 [{agent_name}] {model} ({param_str})")


def setup_file_logging(output_dir: "Path") -> "Path":
    """
    Add file handler to logger to capture all logs to the output directory.
    
    Args:
        output_dir: Path to the output directory (e.g., outputs/briefing_20260124_132153)
        
    Returns:
        Path to the log file
    """
    global _file_handler
    from pathlib import Path
    
    # Remove existing file handler if any
    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()
    
    # Create log file path
    log_file = Path(output_dir) / "pipeline.log"
    
    # Create file handler with same format as console
    _file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    _file_handler.setLevel(logging.DEBUG)  # Capture everything including DEBUG
    _file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Add to root logger only - this captures all logs including briefing, httpx, google_genai
    # Don't add to briefing logger directly to avoid duplicate logs (briefing propagates to root)
    logging.getLogger().addHandler(_file_handler)
    
    logger.info(f"📝 Logging to: {log_file}")
    
    return log_file


def close_file_logging() -> None:
    """Close file handler and clean up."""
    global _file_handler
    
    if _file_handler is not None:
        logging.getLogger().removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None


T = TypeVar('T')


# =============================================================================
# RETRY LOGIC
# =============================================================================

class RetryableError(Exception):
    """Errors that should trigger a retry."""
    pass


class NonRetryableError(Exception):
    """Errors that should NOT trigger a retry."""
    pass


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (including first try).
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff.
        jitter: Whether to add random jitter to delays.
        retryable_exceptions: Tuple of exception types to retry on.
    
    Returns:
        Decorated function with retry logic.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except NonRetryableError:
                    # Don't retry these
                    raise
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Add jitter (±25%)
                    if jitter:
                        delay = delay * (0.75 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable based on common patterns."""
    error_str = str(error).lower()
    
    # Rate limiting
    if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
        return True
    
    # Server errors
    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return True
    
    # Network errors
    if "timeout" in error_str or "connection" in error_str:
        return True
    
    # Resource exhausted
    if "resource exhausted" in error_str or "overloaded" in error_str:
        return True
    
    return False


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

@dataclass
class BriefingMetrics:
    """Metrics collected during a Briefing run."""
    run_id: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Phase timings (seconds)
    phase_1_duration: float = 0.0
    phase_2_duration: float = 0.0
    phase_3_duration: float = 0.0
    phase_4_duration: float = 0.0
    phase_5_duration: float = 0.0
    
    # Counts
    event_count: int = 0
    region_count: int = 0
    storyline_count: int = 0
    black_swan_count: int = 0
    
    # Generation
    word_count: int = 0
    regeneration_attempts: int = 0
    
    # Scores
    judge_score: float = 0.0
    verdict: str = ""
    
    # Errors
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "durations": {
                "phase_1_aggregate": self.phase_1_duration,
                "phase_2_cluster": self.phase_2_duration,
                "phase_3_generate": self.phase_3_duration,
                "phase_4_verify": self.phase_4_duration,
                "phase_5_store": self.phase_5_duration,
                "total": sum([
                    self.phase_1_duration,
                    self.phase_2_duration,
                    self.phase_3_duration,
                    self.phase_4_duration,
                    self.phase_5_duration,
                ]),
            },
            "counts": {
                "events": self.event_count,
                "regions": self.region_count,
                "storylines": self.storyline_count,
                "black_swans": self.black_swan_count,
            },
            "generation": {
                "word_count": self.word_count,
                "regeneration_attempts": self.regeneration_attempts,
            },
            "quality": {
                "judge_score": self.judge_score,
                "verdict": self.verdict,
            },
            "errors": self.errors,
        }


def log_phase_start(phase: int, name: str):
    """Log the start of a pipeline phase."""
    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE {phase}: {name}")
    logger.info(f"{'=' * 60}")


def log_phase_end(phase: int, name: str, duration: float, **metrics):
    """Log the end of a pipeline phase with metrics."""
    metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"Phase {phase} ({name}) completed in {duration:.1f}s | {metric_str}")


# =============================================================================
# CACHING
# =============================================================================

@dataclass
class PipelineCache:
    """
    Cache for expensive computations within a single Briefing run.
    
    Useful during regeneration loops to avoid re-computing embeddings, etc.
    """
    # Phase 1 outputs
    events: list[dict] | None = None
    by_region: dict[str, list[dict]] | None = None
    week_start: datetime | None = None
    week_end: datetime | None = None
    
    # Phase 2 outputs
    embeddings: Any | None = None  # np.ndarray
    cluster_analysis: Any | None = None  # ClusterAnalysis
    themes: dict[str, list] | None = None
    
    # Calendar (Phase 3 input)
    upcoming_events: list | None = None
    
    def clear(self):
        """Clear all cached data."""
        self.events = None
        self.by_region = None
        self.week_start = None
        self.week_end = None
        self.embeddings = None
        self.cluster_analysis = None
        self.themes = None
        self.upcoming_events = None
    
    def has_phase_1(self) -> bool:
        """Check if Phase 1 outputs are cached."""
        return self.events is not None and self.by_region is not None
    
    def has_phase_2(self) -> bool:
        """Check if Phase 2 outputs are cached."""
        return self.cluster_analysis is not None and self.themes is not None


# Global cache instance (reset per run)
_cache = PipelineCache()


def get_cache() -> PipelineCache:
    """Get the global pipeline cache."""
    return _cache


def reset_cache():
    """Reset the global pipeline cache."""
    _cache.clear()
