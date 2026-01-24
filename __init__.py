"""
The Briefing - Intelligence Analysis Pipeline

Multi-agent editorial pipeline that transforms raw event data into 
structured intelligence briefings.

Multi-Agent Pipeline (8 agents):
- Editor: Editorial research & kill decisions
- Architect: Document skeleton & narrative arc
- Analyst: Constraints-of-Thought, Futures Wheel, ACH reasoning
- Structure: Beat sheets and paragraph plans per archetype
- Writer: Chain of Density prose generation
- Stylist: Voice transformation (Economist/Stratfor style)
- Critic: CoVe verification, dual-track feedback routing
- Assembler: Final document assembly with narrative arc

Usage:
    import asyncio
    from run import run_pipeline
    
    result = asyncio.run(run_pipeline(mode="test", dry_run=True))
"""

__version__ = "3.1.0"
