"""
The Briefing Multi-Agent Analysis Engine

7-Agent Editorial Pipeline:
- Analyst: Structured reasoning (Const-o-T, Futures Wheel, ACH)
- Editor: Editorial research and kill/publish decisions
- Architect: Structure Editor decisions into DocumentSkeleton JSON
- Structure: Beat sheets and paragraph plans
- Writer: Dense prose generation following blueprints
- Stylist: Voice transformation (Economist/Stratfor style)
- Critic: CoVe verification, dual-track feedback routing
- Assembler: Final document assembly with narrative arc
"""

from .schemas import (
    AnalystInput,
    AnalystOutput,
    ActorAnalysis,
    ConstraintSet,
    FuturesWheel,
    CompetingHypothesis,
    WriterInput,
    WriterOutput,
)
from .analyst import run_analyst_agent
from .writer import run_writer_agent
from .editor import (
    EditorInput,
    EditorDecisions,
    run_editor_agent,
)
from .architect import (
    ArchitectInput,
    run_architect_agent,
)
from .editorial_utils import (
    calculate_pic_score,
    apply_delta_test,
    analyze_thematic_synchronization,
    calculate_signal_corroboration_score,
)
from .structure import (
    StructureInput,
    run_structure_agent,
    run_structure_agents_parallel,
    create_fallback_blueprint,
    ARCHETYPE_TEMPLATES,
)
from .stylist import (
    StylistInput,
    run_stylist_agent,
    run_stylist_agents_parallel,
    apply_orwell_filter,
    verify_sacred_elements,
    extract_sacred_elements,
)
from .critic import (
    run_cove,
    extract_factual_claims,
    run_content_critic,
    run_style_critic,
    run_writer_loop,
    run_stylist_loop,
    run_content_pipeline,
)

__all__ = [
    # Schemas
    "AnalystInput",
    "AnalystOutput",
    "ActorAnalysis",
    "ConstraintSet",
    "FuturesWheel",
    "CompetingHypothesis",
    "WriterInput",
    "WriterOutput",
    # Analyst
    "run_analyst_agent",
    # Writer
    "run_writer_agent",
    # Editor
    "EditorInput",
    "EditorDecisions",
    "run_editor_agent",
    # Architect
    "ArchitectInput",
    "run_architect_agent",
    # Editorial frameworks
    "calculate_pic_score",
    "apply_delta_test",
    "analyze_thematic_synchronization",
    "calculate_signal_corroboration_score",
    # Structure
    "StructureInput",
    "run_structure_agent",
    "run_structure_agents_parallel",
    "create_fallback_blueprint",
    "ARCHETYPE_TEMPLATES",
    # Stylist
    "StylistInput",
    "run_stylist_agent",
    "run_stylist_agents_parallel",
    "apply_orwell_filter",
    "verify_sacred_elements",
    "extract_sacred_elements",
    # Critic (split workflow)
    "run_cove",
    "extract_factual_claims",
    "run_content_critic",
    "run_style_critic",
    "run_writer_loop",
    "run_stylist_loop",
    "run_content_pipeline",
]
