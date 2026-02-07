"""
The Briefing - Configuration

Central configuration with production/test mode support.
Production mode uses Gemini 3.0 Pro for maximum quality.
Test mode uses Gemini 3.0 Flash for cheap pipeline validation.

Usage:
    from config import get_config
    
    config = get_config()  # Returns BriefingConfig based on BRIEFING_MODE env var
    
    # Access model names
    config.models.analyst  # Returns appropriate model for mode
    
    # Override mode
    config = get_config(mode="test")

Environment variables use BRIEFING_* prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _env_int(key: str, default: int) -> int:
    """Get int from environment variable or use default."""
    val = os.getenv(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _env_float(key: str, default: float | None) -> float | None:
    """Get float from environment variable or use default."""
    val = os.getenv(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment variable or use default."""
    val = os.getenv(key)
    if val is not None:
        return val.lower() in ("true", "1", "yes")
    return default


def _env_str(key: str, default: str | None) -> str | None:
    """Get string from environment variable or use default."""
    return os.getenv(key, default)


def _env_int_or_none(key: str, default: int | None = None) -> int | None:
    """Get int from environment variable, or None if not set."""
    val = os.getenv(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


@dataclass
class ModelConfig:
    """Model configuration for a specific run mode."""

    embedding: str
    theme: str
    editor: str
    architect: str
    analyst: str
    structure: str
    writer: str
    stylist: str
    critic: str
    image: str


# Production models: Gemini 3 Pro Preview for maximum quality
PRODUCTION_MODELS = ModelConfig(
    embedding=os.getenv("BRIEFING_EMBEDDING_MODEL", "gemini-embedding-001"),
    theme=os.getenv("BRIEFING_THEME_MODEL", "gemini-3-pro-preview"),
    editor=os.getenv("BRIEFING_EDITOR_MODEL", "gemini-3-pro-preview"),
    architect=os.getenv("BRIEFING_ARCHITECT_MODEL", "gemini-3-pro-preview"),
    analyst=os.getenv("BRIEFING_ANALYST_MODEL", "gemini-3-pro-preview"),
    structure=os.getenv("BRIEFING_STRUCTURE_MODEL", "gemini-3-pro-preview"),
    writer=os.getenv("BRIEFING_WRITER_MODEL", "gemini-3-pro-preview"),
    stylist=os.getenv("BRIEFING_STYLIST_MODEL", "gemini-3-pro-preview"),
    critic=os.getenv("BRIEFING_CRITIC_MODEL", "gemini-3-pro-preview"),
    image=os.getenv("BRIEFING_IMAGE_MODEL", "gemini-2.5-flash-image"),
)

# Test models: Gemini 3 Flash Preview for near-production quality iteration
TEST_MODELS = ModelConfig(
    embedding=os.getenv("BRIEFING_EMBEDDING_MODEL", "gemini-embedding-001"),
    theme=os.getenv("BRIEFING_THEME_MODEL", "gemini-3-flash-preview"),
    editor=os.getenv("BRIEFING_EDITOR_MODEL", "gemini-3-flash-preview"),
    architect=os.getenv("BRIEFING_ARCHITECT_MODEL", "gemini-3-flash-preview"),
    analyst=os.getenv("BRIEFING_ANALYST_MODEL", "gemini-3-flash-preview"),
    structure=os.getenv("BRIEFING_STRUCTURE_MODEL", "gemini-3-flash-preview"),
    writer=os.getenv("BRIEFING_WRITER_MODEL", "gemini-3-flash-preview"),
    stylist=os.getenv("BRIEFING_STYLIST_MODEL", "gemini-3-flash-preview"),
    critic=os.getenv("BRIEFING_CRITIC_MODEL", "gemini-3-flash-preview"),
    image=os.getenv("BRIEFING_IMAGE_MODEL", "gemini-2.5-flash-image"),
)


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================


@dataclass
class BriefingConfig:
    """
    The Briefing - Pipeline Configuration.
    
    Production mode optimizes for QUALITY with no artificial limits.
    Test mode optimizes for SPEED/COST to validate pipeline structure.
    
    Environment variables use BRIEFING_* prefix.
    """
    
    # =========================================================================
    # MODE SELECTION
    # =========================================================================

    mode: Literal["production", "test"] = field(
        default_factory=lambda: _env_str("BRIEFING_MODE", "production")
    )

    @property
    def models(self) -> ModelConfig:
        """Get model configuration for current mode."""
        return PRODUCTION_MODELS if self.mode == "production" else TEST_MODELS
    
    # =========================================================================
    # API KEYS
    # =========================================================================
    
    gemini_api_key: str = field(
        default_factory=lambda: _env_str("GEMINI_API_KEY", "")
    )
    tavily_api_key: str = field(
        default_factory=lambda: _env_str("TAVILY_API_KEY", "")
    )
    
    # =========================================================================
    # SUPABASE (ATLAS)
    # =========================================================================
    
    supabase_url: str = field(
        default_factory=lambda: _env_str("SUPABASE_URL", "")
    )
    supabase_service_key: str = field(
        default_factory=lambda: _env_str("SUPABASE_SERVICE_KEY", "")
    )
    
    # =========================================================================
    # R2 STORAGE
    # =========================================================================

    r2_access_key_id: str = field(
        default_factory=lambda: _env_str("R2_ACCESS_KEY_ID", "")
    )
    r2_secret_access_key: str = field(
        default_factory=lambda: _env_str("R2_SECRET_ACCESS_KEY", "")
    )
    r2_endpoint_url: str = field(
        default_factory=lambda: _env_str("R2_ENDPOINT_URL", "")
    )
    r2_bucket_name: str = field(
        default_factory=lambda: _env_str("R2_BUCKET_NAME", "")
    )
    r2_public_url: str = field(
        default_factory=lambda: _env_str("R2_PUBLIC_URL", "")
    )
    
    # =========================================================================
    # EMBEDDING SETTINGS
    # =========================================================================
    
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("BRIEFING_EMBEDDING_BATCH_SIZE", 100)
    )
    embedding_dimensions: int = field(
        default_factory=lambda: _env_int("BRIEFING_EMBEDDING_DIMENSIONS", 768)
    )
    
    # =========================================================================
    # CLUSTERING SETTINGS
    # =========================================================================
    
    hdbscan_min_cluster_size: int = field(
        default_factory=lambda: _env_int("BRIEFING_HDBSCAN_MIN_CLUSTER_SIZE", 3)
    )
    hdbscan_min_samples: int = field(
        default_factory=lambda: _env_int("BRIEFING_HDBSCAN_MIN_SAMPLES", 2)
    )
    black_swan_threshold: int = field(
        default_factory=lambda: _env_int("BRIEFING_BLACK_SWAN_THRESHOLD", 8)
    )
    min_region_events: int = field(
        default_factory=lambda: _env_int("BRIEFING_MIN_REGION_EVENTS", 3)
    )
    
    # Theme labeling (LLM call during clustering)
    theme_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_THEME_TEMPERATURE", None)
    )
    theme_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_THEME_MAX_TOKENS", None)
    )
    
    # =========================================================================
    # THEME EXTRACTION
    # =========================================================================
    
    max_storylines_per_region: int = field(
        default_factory=lambda: _env_int("BRIEFING_MAX_STORYLINES_PER_REGION", 5)
    )
    max_events_per_storyline: int = field(
        default_factory=lambda: _env_int("BRIEFING_MAX_EVENTS_PER_STORYLINE", 10)
    )
    max_featured_candidates: int = field(
        default_factory=lambda: _env_int("BRIEFING_MAX_FEATURED_CANDIDATES", 10)
    )
    
    # =========================================================================
    # EDITORIAL SETTINGS
    # =========================================================================
    
    # Target total word count for the briefing
    total_word_budget: int = field(
        default_factory=lambda: _env_int("BRIEFING_TOTAL_WORDS", 2000)
    )
    
    # Style preset for prose generation
    # Options: "economist" (The Economist style), "stratfor" (Stratfor analysis), "mixed" (both)
    style_preset: str = field(
        default_factory=lambda: _env_str("BRIEFING_STYLE_PRESET", "mixed")
    )
    
    # =========================================================================
    # MULTI-AGENT PIPELINE (v2)
    # =========================================================================
    
    # Max retry attempts for Writer/Stylist when Critic fails them
    max_retry_attempts: int = field(
        default_factory=lambda: _env_int("BRIEFING_MAX_RETRY_ATTEMPTS", 5)
    )
    
    # Critic pass threshold (0-100)
    critic_pass_threshold: int = field(
        default_factory=lambda: _env_int("BRIEFING_CRITIC_PASS_THRESHOLD", 70)
    )
    
    # Max critic loop iterations (total Writer + Stylist retries before giving up)
    max_critic_loops: int = field(
        default_factory=lambda: _env_int("BRIEFING_MAX_CRITIC_LOOPS", 10)
    )
    
    # Parallel analyst processing
    max_concurrent_analysts: int = field(
        default_factory=lambda: _env_int("BRIEFING_MAX_CONCURRENT_ANALYSTS", 5)
    )
    
    # =========================================================================
    # PER-AGENT GENERATION SETTINGS
    # =========================================================================
    
    # Editor agent (Phase 3A: Editorial research + decisions)
    editor_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_EDITOR_TEMPERATURE", None)
    )
    editor_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_EDITOR_MAX_TOKENS", None)
    )
    editor_thinking_level: str | None = field(
        default_factory=lambda: _env_str("BRIEFING_EDITOR_THINKING", None)
    )
    editor_max_tool_rounds: int = field(
        default_factory=lambda: _env_int("BRIEFING_EDITOR_TOOL_ROUNDS", 5)
    )
    
    # Architect agent (Phase 3B: Structure into JSON + synthesize narrative arc)
    architect_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_ARCHITECT_TEMPERATURE", None)
    )
    architect_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_ARCHITECT_MAX_TOKENS", None)
    )
    architect_thinking_level: str | None = field(
        default_factory=lambda: _env_str("BRIEFING_ARCHITECT_THINKING", None)
    )
    architect_formatter_thinking_level: str | None = field(
        default_factory=lambda: _env_str("BRIEFING_ARCHITECT_FORMATTER_THINKING", None)
    )
    
    # Analyst agent (Phase 4: Deep analysis)
    analyst_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_ANALYST_TEMPERATURE", None)
    )
    analyst_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_ANALYST_MAX_TOKENS", None)
    )
    analyst_thinking_level: str | None = field(
        default_factory=lambda: _env_str("BRIEFING_ANALYST_THINKING", None)
    )
    analyst_max_tool_rounds: int = field(
        default_factory=lambda: _env_int("BRIEFING_ANALYST_TOOL_ROUNDS", 10)
    )
    
    # Structure agent (Phase 5: Beat sheets)
    structure_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_STRUCTURE_TEMPERATURE", None)
    )
    structure_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_STRUCTURE_MAX_TOKENS", None)
    )
    
    # Writer agent (Phase 6: Prose generation)
    writer_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_WRITER_TEMPERATURE", None)
    )
    writer_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_WRITER_MAX_TOKENS", None)
    )
    writer_thinking_level: str | None = field(
        default_factory=lambda: _env_str("BRIEFING_WRITER_THINKING", None)
    )
    writer_cod_max_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_WRITER_COD_MAX_TOKENS", None)
    )
    writer_cod_thinking_level: str | None = field(
        default_factory=lambda: _env_str("BRIEFING_WRITER_COD_THINKING", None)  # None = model defaults
    )
    
    # Stylist agent (Phase 6b: Voice transformation)
    stylist_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_STYLIST_TEMPERATURE", None)
    )
    stylist_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_STYLIST_MAX_TOKENS", None)
    )
    
    # Critic agent (Phase 7: Quality control)
    critic_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_CRITIC_TEMPERATURE", None)
    )
    critic_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_CRITIC_MAX_TOKENS", None)
    )
    critic_devils_advocate_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_CRITIC_DEVILS_ADVOCATE_TEMPERATURE", None)
    )
    critic_devils_advocate_max_tokens: int = field(
        default_factory=lambda: _env_int("BRIEFING_CRITIC_DEVILS_ADVOCATE_MAX_TOKENS", 1500)
    )
    
    # =========================================================================
    # RETRY SETTINGS
    # =========================================================================
    
    retry_initial_delay: float = field(
        default_factory=lambda: _env_float("BRIEFING_RETRY_INITIAL_DELAY", 2.0)
    )
    retry_max_delay: float = field(
        default_factory=lambda: _env_float("BRIEFING_RETRY_MAX_DELAY", 60.0)
    )
    
    # =========================================================================
    # CROSS-REGIONAL & CALENDAR
    # =========================================================================
    
    cross_regional_similarity_threshold: float = field(
        default_factory=lambda: _env_float("BRIEFING_CROSS_REGIONAL_SIMILARITY", 0.7)
    )
    
    # =========================================================================
    # IMAGE GENERATION
    # =========================================================================
    
    max_images_per_briefing: int = field(
        default_factory=lambda: _env_int("BRIEFING_MAX_IMAGES", 3)
    )
    image_temperature: float | None = field(
        default_factory=lambda: _env_float("BRIEFING_IMAGE_TEMPERATURE", None)
    )
    image_max_output_tokens: int | None = field(
        default_factory=lambda: _env_int_or_none("BRIEFING_IMAGE_MAX_TOKENS", None)
    )
    
    # =========================================================================
    # POST-INIT ADJUSTMENTS
    # =========================================================================

    def __post_init__(self):
        """Adjust settings based on mode."""
        if self.mode == "test":
            # Test mode: Faster iteration with lower quality thresholds
            self.critic_pass_threshold = 50
            self.max_retry_attempts = 1  # Faster iteration in test mode
            self.max_critic_loops = 3  # Limit total critic cycles
            
            # All other params (temperature, max_tokens, thinking_level) already default to None
            # which lets models use their own defaults - perfect for test mode

    def __repr__(self) -> str:
        """Pretty print config (hide API keys)."""
        lines = [f"BriefingConfig (mode={self.mode}):"]
        lines.append(f"  Models:")
        lines.append(f"    editor: {self.models.editor}")
        lines.append(f"    architect: {self.models.architect}")
        lines.append(f"    analyst: {self.models.analyst}")
        lines.append(f"    structure: {self.models.structure}")
        lines.append(f"    writer: {self.models.writer}")
        lines.append(f"    stylist: {self.models.stylist}")
        lines.append(f"    critic: {self.models.critic}")
        lines.append(f"    theme: {self.models.theme}")
        lines.append(f"    image: {self.models.image}")
        lines.append(f"  Settings:")
        for key, value in self.__dict__.items():
            if key == "mode":
                continue
            if "key" in key.lower() or "webhook" in key.lower() or "secret" in key.lower():
                value = "***" if value else "(not set)"
            lines.append(f"    {key}: {value}")
        return "\n".join(lines)


# =============================================================================
# GLOBAL CONFIG ACCESS
# =============================================================================

_config: BriefingConfig | None = None


def get_config(mode: str | None = None) -> BriefingConfig:
    """
    Get the global configuration.

    Args:
        mode: Override mode ("production" or "test"). If None, uses BRIEFING_MODE env var.
              When mode is provided, updates the global config for all subsequent calls.

    Returns:
        BriefingConfig instance
    """
    global _config

    if mode is not None:
        # Explicit mode override - create new config and set as global
        _config = BriefingConfig(mode=mode)
        return _config

    if _config is None:
        _config = BriefingConfig()

    return _config


def reset_config():
    """Reset the global config (useful for testing)."""
    global _config
    _config = None


# Legacy compatibility
config = get_config()


def print_config():
    """Print current configuration."""
    print(get_config())
