"""
The Briefing Example Bank.

This module provides curated examples for few-shot learning,
particularly for style transformation by the Stylist agent.

Usage:
    from examples import get_style_examples, get_negative_constraints
    
    examples = get_style_examples(topic="Elections", tone="Analytical")
    constraints = get_negative_constraints()
"""

from .economist_style import ECONOMIST_EXAMPLES, STRATFOR_EXAMPLES
from .negative_constraints import NEGATIVE_CONSTRAINTS, CLICHE_LIST
from .retriever import ExampleRetriever, get_style_examples

__all__ = [
    "ECONOMIST_EXAMPLES",
    "STRATFOR_EXAMPLES", 
    "NEGATIVE_CONSTRAINTS",
    "CLICHE_LIST",
    "ExampleRetriever",
    "get_style_examples",
]
