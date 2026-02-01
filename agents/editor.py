"""
Editor Agent - Editorial Research & Decision Making

The Editor is the first editorial agent in the pipeline. It:
1. Reviews cluster data (headlines, severity, event counts)
2. Uses deep thinking (24K tokens) to reason about significance
3. Researches context via Tavily as needed
4. Makes kill/publish decisions using PIC Matrix and Delta Test
5. Determines organizing principle (REGIONAL/THEMATIC/HYBRID)
6. Identifies hub mechanism (if THEMATIC)

Output is EditorDecisions - a text-based editorial brief that the Architect
will structure into DocumentSkeleton JSON and synthesize narrative arc.

Key Decision Frameworks:
- PIC Matrix: Probability × Impact × Confidence scoring
- Delta Test: "Does this change the 6-month forecast?"
- Synchronization Threshold: 60% shared driver → THEMATIC organization
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from config import get_config
from utils import logger, with_retry, log_model_config
from tools import get_tavily_search_tool, execute_tool_call

if TYPE_CHECKING:
    pass


# =============================================================================
# TRANSITION VALIDATION
# =============================================================================

def validate_transition_quality(transition: str, section_a: str, section_b: str) -> tuple[bool, str, str]:
    """
    Validate that a transition shows real connection, not geographic filler.
    
    Returns:
        (is_quality, connection_type, feedback)
    """
    import re
    
    if transition == "HARD_BREAK":
        return True, "HARD_BREAK", "Explicit break - acceptable"
    
    text_lower = transition.lower()
    
    # Red flags (geographic/temporal filler)
    filler_patterns = {
        r'\bwe turn to\b': "Geographic filler: 'we turn to'",
        r'\bturning to\b': "Geographic filler: 'turning to'",
        r'\bshifting? focus\b': "Geographic filler: 'shifting focus'",
        r'\bmeanwhile\b': "Temporal filler: 'meanwhile'",
        r'\balso this week\b': "Temporal filler: 'also this week'",
        r'\bin another region\b': "Geographic filler: 'in another region'",
        r'\bfinally,? (?:we|from)\b': "Sequential filler: 'finally, we...'",
        r'\blooking at\b': "Geographic filler: 'looking at'",
        r'\bfrom .+, we (?:turn|shift|look)\b': "Geographic movement pattern",
    }
    
    for pattern, issue in filler_patterns.items():
        if re.search(pattern, text_lower):
            return False, "FILLER", issue
    
    # Green flags (causal connection types)
    connection_types = {
        r'\bthe same .+ that\b': "MECHANISM",
        r'\bestablishes? (?:the |a )?(?:precedent|pattern|condition)\b': "CONSEQUENCE",
        r'\bevery .+ spent on .+ is .+ not\b': "TRADEOFF",
        r'\bif .+ shows .+, .+ reveals?\b': "CONTRAST",
        r'\b(?:because|therefore|thus|consequently)\b': "CAUSAL",
        r'\b(?:enables?|constrains?|drives?|compels?|forces?)\b': "CAUSAL",
        r'\bcreates? (?:the |a )?(?:condition|precedent|opening|vacuum)\b': "CONSEQUENCE",
    }
    
    for pattern, conn_type in connection_types.items():
        if re.search(pattern, text_lower):
            return True, conn_type, f"Shows {conn_type} connection"
    
    # No clear causal language found
    return False, "WEAK", "Lacks causal language - may be descriptive rather than connective"


def validate_all_transitions(transitions: dict[str, str]) -> dict:
    """
    Validate all transitions in a skeleton.
    
    Returns summary stats and warnings.
    """
    stats = {
        'total': len(transitions),
        'hard_breaks': 0,
        'causal': 0,
        'filler': 0,
        'weak': 0,
        'by_type': {},
        'warnings': [],
    }
    
    for key, text in transitions.items():
        section_a, section_b = key.split(' → ') if ' → ' in key else ('?', '?')
        is_quality, conn_type, feedback = validate_transition_quality(text, section_a, section_b)
        
        # Track stats
        stats['by_type'][conn_type] = stats['by_type'].get(conn_type, 0) + 1
        
        if conn_type == "HARD_BREAK":
            stats['hard_breaks'] += 1
        elif is_quality:
            stats['causal'] += 1
        elif conn_type == "FILLER":
            stats['filler'] += 1
            stats['warnings'].append(f"  ⚠️ '{key}': {feedback}")
        else:
            stats['weak'] += 1
            stats['warnings'].append(f"  💭 '{key}': {feedback}")
    
    return stats


# =============================================================================
# EDITOR INPUT
# =============================================================================

@dataclass
class EditorInput:
    """
    Input data for the Editor Agent.
    
    The Editor reviews cluster data (not full analysis) to make editorial decisions.
    """
    
    # Primary input: thematic clusters from Phase 2
    thematic_clusters: list[dict]  # ThematicCluster.to_dict() results
    hub_candidates: list[dict]  # Potential hub themes (touch 3+ regions)
    total_word_budget: int  # Target total words for briefing
    recommended_organization: str = "REGIONAL"  # Clustering's recommendation
    
    # Context
    cross_regional_connections: list[str] = None  # Connections identified in clustering
    calendar_events: list[dict] = None  # Upcoming events for context
    
    # Previous edition context (for editorial continuity)
    previous_edition: dict | None = None  # {skeleton, date, run_id} from find_previous_edition()
    
    def __post_init__(self):
        if self.cross_regional_connections is None:
            self.cross_regional_connections = []
        if self.calendar_events is None:
            self.calendar_events = []


# =============================================================================
# EDITOR OUTPUT
# =============================================================================

@dataclass
class EditorDecisions:
    """
    Output from the Editor agent.
    
    This is a text-based editorial brief that captures kill/publish decisions,
    narrative arc, and organizing principles. The Architect will structure
    this into DocumentSkeleton JSON.
    """
    
    editorial_brief: str  # The Editor's complete reasoning and decisions
    conversation_history: list  # Full message history for Architect context
    
    def to_dict(self) -> dict:
        return {
            "editorial_brief": self.editorial_brief,
            "message_count": len(self.conversation_history),
        }


# =============================================================================
# EDITOR PROMPT HELPERS
# =============================================================================

def _format_thematic_clusters(
    hub_candidates: list[dict] | None,
    thematic_clusters: list[dict] | None,
) -> str:
    """Format thematic cluster data for the Editor prompt."""
    if not thematic_clusters and not hub_candidates:
        return "No thematic clusters detected. Use REGIONAL organization."
    
    lines = []
    
    # Hub candidates (most important)
    if hub_candidates:
        lines.append("### 🎯 HUB CANDIDATES (themes spanning 3+ regions)")
        lines.append("")
        for hub in hub_candidates[:3]:  # Top 3
            label = hub.get("theme_label", f"Theme {hub.get('cluster_id', '?')}")
            regions = ", ".join(hub.get("regions_touched", []))
            event_count = hub.get("event_count", 0)
            severity = hub.get("avg_severity", 0)
            lines.append(f"- **{label}**")
            lines.append(f"  - Regions: {regions}")
            lines.append(f"  - Events: {event_count} (avg severity: {severity:.1f})")
            lines.append("")
    else:
        lines.append("### No Hub candidates (no theme spans 3+ regions with high severity)")
        lines.append("")
    
    # Other thematic clusters
    if thematic_clusters:
        non_hub = [c for c in thematic_clusters if not c.get("is_hub_candidate")][:3]
        if non_hub:
            lines.append("### Other Thematic Clusters")
            lines.append("")
            for cluster in non_hub:
                label = cluster.get("theme_label", f"Theme {cluster.get('cluster_id', '?')}")
                regions = ", ".join(cluster.get("regions_touched", []))
                lines.append(f"- {label}: spans {regions}")
            lines.append("")
    
    return "\n".join(lines)


def _format_previous_edition(previous: dict) -> str:
    """
    Format previous edition skeleton into Editor context.
    
    Args:
        previous: Dict with 'skeleton', 'date', 'run_id' from find_previous_edition()
    
    Returns:
        Markdown-formatted context for the Editor prompt
    """
    skeleton = previous.get("skeleton", {})
    date = previous.get("date", "last week")
    
    lines = [
        f"## 📚 PREVIOUS EDITION ({date})",
        "",
    ]
    
    # Hub mechanism
    hub_mech = skeleton.get("hub_mechanism")
    if hub_mech:
        lines.append(f"**Hub Mechanism:** {hub_mech}")
        hub_exp = skeleton.get("hub_explanation", "")
        if hub_exp:
            lines.append(f"> {hub_exp[:300]}...")
        lines.append("")
    
    # Narrative arc
    arc = skeleton.get("narrative_arc", "")
    if arc:
        lines.append(f"**Narrative Arc:** {arc}")
        lines.append("")
    
    # Published stories table
    lines.append("**PUBLISHED STORIES:**")
    lines.append("")
    lines.append("| Story | Treatment | Cluster ID |")
    lines.append("|-------|-----------|------------|")
    
    # Featured
    featured = skeleton.get("featured", {})
    if featured:
        headline = featured.get("headline", "?")
        cluster_id = featured.get("source_cluster_id", "?")
        lines.append(f"| {headline} | FEATURED | {cluster_id} |")
    
    # Sections
    for section in skeleton.get("sections", []):
        headline = section.get("headline", "?")
        treatment = section.get("treatment", "?")
        cluster_id = section.get("source_cluster_id", "?")
        lines.append(f"| {headline} | {treatment} | {cluster_id} |")
    
    lines.append("")
    
    # Quick hits
    quick_hits = skeleton.get("quick_hits", [])
    if quick_hits:
        lines.append("**Quick Hits:** " + ", ".join(
            qh.get("headline", qh.get("region", "?")) for qh in quick_hits
        ))
        lines.append("")
    
    # Killed clusters - especially HOLD_FOR_NEXT_WEEK
    killed = skeleton.get("killed", [])
    if killed:
        held = [k for k in killed if k.get("reason") == "HOLD_FOR_NEXT_WEEK"]
        other_killed = len(killed) - len(held)
        
        if held:
            lines.append("**⚠️ HELD FOR THIS WEEK:**")
            for k in held:
                cluster_id = k.get("region", k.get("cluster_id", "?"))
                note = k.get("brief_explanation", "")
                pic = k.get("pic_score", 0)
                lines.append(f"- {cluster_id} (PIC: {pic}): {note}")
            lines.append("")
        
        if other_killed:
            lines.append(f"**Killed last week:** {other_killed} clusters")
            lines.append("")
    
    # Editorial guidance
    lines.extend([
        "---",
        "",
        "**CONTINUITY GUIDANCE:**",
        "- **Same story with updates**: Lead with what's NEW, one-sentence recap max",
        "- **Story resolved**: Quick Hit noting resolution",
        "- **Story unchanged**: KILL (readers already know)",
        "- **HELD clusters**: Consider covering this week - they were deferred, not killed",
        "- **Predictions**: Note if any previous forecasts proved correct/incorrect",
        "",
    ])
    
    return "\n".join(lines)


def _build_editor_prompt(input_data: EditorInput) -> str:
    """
    Build the prompt for the Editor agent.
    
    Follows Gemini prompt guide best practices:
    - XML structure separating context from task
    - Clear constraints
    - Knowledge cutoff notice
    """
    
    # Build cluster summaries
    cluster_summaries = []
    for cluster in input_data.thematic_clusters:
        cluster_id = cluster.get("cluster_id", "?")
        label = cluster.get("theme_label", "")  # Optional display name
        events = cluster.get("events", [])
        
        # Rank events by credibility (source count + severity)
        ranked_events = sorted(
            events,
            key=lambda e: (
                len(e.get("sources", [])) if isinstance(e.get("sources"), list) else 1,
                e.get("severity", 0)
            ),
            reverse=True
        )
        
        # Extract top headlines
        headlines = []
        for evt in ranked_events[:5]:
            title = evt.get("title", "")[:120]
            summary = evt.get("summary", "")[:200] if evt.get("summary") else ""
            source_count = len(evt.get("sources", [])) if isinstance(evt.get("sources"), list) else 1
            severity = evt.get("severity", 0)
            
            headline_entry = f"{title}"
            if summary:
                headline_entry += f" — {summary}"
            headline_entry += f" [sources: {source_count}, severity: {severity}]"
            headlines.append(headline_entry)
        
        # Build display name: "Cluster ID (label)" or just "Cluster ID"
        display_name = f"cluster_{cluster_id}"
        if label:
            display_name = f"{label} (cluster_{cluster_id})"
        
        cluster_summaries.append({
            "cluster_id": f"cluster_{cluster_id}",  # Canonical ID for matching
            "display_name": display_name,  # For Editor's reference
            "event_count": cluster.get("event_count", len(events)),
            "avg_severity": cluster.get("avg_severity", 0),
            "regions_touched": cluster.get("regions_touched", []),
            "is_hub_candidate": cluster.get("is_hub_candidate", False),
            "top_headlines": headlines,
        })
    
    # Format calendar events
    calendar_summary = []
    for event in (input_data.calendar_events or [])[:10]:
        title = event.get("title") or event.get("event", "")
        region = event.get("region", "")
        if not region and "relates_to" in event:
            relates = event.get("relates_to", [])
            region = relates[0] if relates else "Global"
        
        calendar_summary.append({
            "title": title,
            "date": str(event.get("date", "")),
            "region": region or "Global",
        })
    
    # Build hub candidate summary
    hub_summary = []
    for hub in input_data.hub_candidates:
        hub_summary.append({
            "theme": hub.get("theme_label", "Unknown"),
            "regions": hub.get("regions_touched", []),
            "event_count": hub.get("event_count", 0),
            "avg_severity": hub.get("avg_severity", 0),
        })
    
    # Build previous edition context if available
    previous_context = ""
    if input_data.previous_edition:
        previous_context = _format_previous_edition(input_data.previous_edition) + "\n---\n\n"
    
    prompt = f"""{previous_context}## THEMATIC CLUSTERS TO REVIEW

You have received {len(cluster_summaries)} thematic clusters from semantic analysis.
Each cluster groups related events across regions. Review each and make editorial decisions.

NOTE: You are working from raw event data (headlines, severity, counts), not analyst interpretations. 
Base your decisions on:
- Event volume and severity
- Multi-source corroboration (more sources = more credible)
- Cross-regional significance
- Whether this changes the 6-month geopolitical forecast

```json
{json.dumps(cluster_summaries, indent=2)}
```

## CROSS-REGIONAL CONNECTIONS IDENTIFIED

{json.dumps(input_data.cross_regional_connections, indent=2) if input_data.cross_regional_connections else "None identified"}

## THEMATIC CLUSTERING ANALYSIS (from global event clustering)

**Clustering Recommendation:** {input_data.recommended_organization}

{_format_thematic_clusters(input_data.hub_candidates, input_data.thematic_clusters)}

If a strong Hub theme exists (touches 4+ regions with high-severity events):
- Use THEMATIC organization with Hub-and-Spoke structure
- **The Hub:** The central mechanism
- **The Spokes:** Regional manifestations

If weaker themes (2-3 regions):
- Consider HYBRID: thematic opening, then regional sections

If no strong themes:
- Use REGIONAL organization

## UPCOMING CALENDAR EVENTS

```json
{json.dumps(calendar_summary, indent=2) if calendar_summary else "None"}
```

## YOUR TASK

1. Score each cluster using the PIC Matrix
2. Apply the Delta Test
3. Decide: KILL / HOLD / PUBLISH for each cluster
4. Select one cluster as FEATURED
5. Assign story archetypes to publishable clusters
6. Assign treatment levels (FULL, SHORT, QUICK_HIT)
7. Determine organizing principle (REGIONAL / THEMATIC / HYBRID)
8. If THEMATIC: Identify the hub mechanism (abstract theme, not a region)
9. Group sections logically
10. Plan transitions between sections

NOTE: The Architect will synthesize the narrative arc based on your decisions.

## WORD BUDGET

Total budget: {input_data.total_word_budget} words

Allocate wisely:
- Featured: ~500 words
- Full sections: 300-400 each
- Short sections: 150-200 each
- Quick Hits: ~100 total

## EXPECTED EDITORIAL DECISIONS

For each cluster, specify:
- **Treatment**: FEATURED / FULL / SHORT / SLEEPER / QUICK_HIT / KILL
- **If publishing**: headline, angle type, archetype, word target, rationale
- **If killing**: reason (NOTHING_NEW, BELOW_THRESHOLD, REDUNDANT, HOLD_FOR_NEXT_WEEK), PIC score

Also define:
- **Organizing principle**: REGIONAL / THEMATIC / HYBRID
- **Hub mechanism** (if THEMATIC): The ABSTRACT mechanism (e.g., "Economic Coercion as Statecraft" NOT "Trump's Trade Policy")
- **Hub explanation** (if THEMATIC): 2-3 sentences explaining the mechanism
- **Section groups**: 2-4 logical groups with evocative titles
- **Transitions**: For each adjacent section pair
  - State the CONNECTION TYPE (MECHANISM, CONSEQUENCE, TRADEOFF, CONTRAST, or HARD_BREAK)
  - Write the transition using the pattern for that type

NOTE: The Architect will synthesize the narrative arc based on your decisions and hub mechanism.

Use search_web tool to verify significance when unsure.
"""
    
    return prompt


# =============================================================================
# EDITOR SYSTEM PROMPT
# =============================================================================

EDITOR_SYSTEM_PROMPT = """
<role>
You are the Editor-in-Chief of a premium weekly intelligence briefing.
Your job is EDITORIAL STRATEGY, not writing.
</role>

<constraints>
1. Not every region deserves coverage every week
2. A briefing that covers everything covers nothing well
3. Be RUTHLESS about editorial focus: "If everything is important, nothing is important"
4. Every decision needs clear reasoning - no hedging on kill decisions
5. Limit published sections to maximum of 10 (featured + sections + quick_hits)
</constraints>

<decisions>
You decide:
1. What stories deserve space (PUBLISH) vs what to cut (KILL)
2. How much space each story gets (FULL, SHORT, QUICK HIT)
3. The order and grouping of sections
4. The narrative that ties the week together
</decisions>

<editorial_philosophy>

THE NUB (The Economist Editorial Philosophy):
For each story, identify THE NUB — not what happened, but what makes it interesting.
The Nub is the tension, contradiction, or intellectual puzzle that justifies the reader's time.

Angle Types (assign one to each publishable story):
- MIND_CLEARING: The topic is messy and confusing. Your angle untangles it and identifies the ONE variable that actually matters.
  Example: "Forget the rhetoric—this is really about port access."
- MISSING_THE_POINT: Conventional coverage focuses on X, but the real story is Y.
  Example: "Everyone's watching the election; the bond market is the story."
- UNINTENDED_CONSEQUENCE: A policy meant to achieve X is actually causing Y.
  Example: "Sanctions designed to isolate Moscow are accelerating de-dollarization."
- CONSTRAINT_REVEALED: Events expose a structural limit on an actor's options.
  Example: "Germany's energy dependence was always the EU's Achilles heel."
- TRAJECTORY_SHIFT: Something changed direction this week.
  Example: "After years of hedging, Ankara has picked a side."

</editorial_philosophy>

<scoring_frameworks>

NEWS VALUE SCORING (for breaking ties):
- IMPACT (1-10): Magnitude of consequences (lives, dollars, territory)
- TIMELINESS (1-10): Urgency—must readers know this NOW?
- CONFLICT (1-10): Clear opposing forces create reader engagement
- NOVELTY (1-10): The "man bites dog" factor—is this an anomaly?
- PROMINENCE (1-10): Major powers involved? Named leaders?
Higher total score wins space.

PIC MATRIX (for kill decisions):
Score each cluster on three dimensions (1-5 each):
- Probability (P): How likely to have lasting consequences?
  5=Near-certain structural shift, 3=Probable but contingent, 1=Unlikely to matter in 30 days
- Impact (I): How significant are potential consequences?
  5=Changes regional balance of power, 3=Affects bilateral relationships, 1=Localized effects
- Confidence (C): How confident in the assessment?
  5=Multiple corroborating sources, 3=Reasonable inference, 1=Speculative
Score = (P × I × C) / 5
Thresholds: <5=KILL, 5-15=HOLD (Quick Hit/Sleeper), >15=PUBLISH

DELTA TEST:
Ask: "Does this update alter the 6-month forecast for the region?"
- YES → Definitely PUBLISH
- NO but novel → Consider as SLEEPER
- NO and routine → KILL

</scoring_frameworks>

<organizing_principles>

Count how many publishable stories share a common causal driver:
- ≥60% share a driver → THEMATIC organization
  Example: "American retrenchment creates openings in three regions"
- <60% share a driver → REGIONAL organization (standard geographic sections)
- HYBRID: Start with thematic hook, then regional sections

THEMATIC MODE REQUIREMENTS:
When you choose THEMATIC, you MUST also define:
- hub_mechanism: The ABSTRACT mechanism connecting regions (NOT actor-specific!)
  ✅ Good (abstract mechanisms): "Economic Coercion as Statecraft", "Unilateral Action Replacing Consensus", "Civilian Infrastructure as Battlefield"
  ❌ Bad (actor-specific): "Trump's Foreign Policy", "US Trade War", "Netanyahu's Strategy"
  
  ABSTRACTION TEST: Could this pattern exist without [specific actor]?
  - "Trump's contradictions" → NO (too specific) ❌
  - "Coercion replacing consensus" → YES (universal) ✅

- hub_explanation: 2-3 sentences explaining the MECHANISM itself (not a story summary)
- hub_angle: What angle makes this mechanism interesting (MISSING_THE_POINT, etc.)

The hub_mechanism should be a FORCE/PATTERN that manifests across multiple regions.
Think: "What underlying mechanism drives multiple stories?" Focus on HOW, not WHO.

</organizing_principles>

<story_archetypes>
Assign each publishable story one archetype:
- CRISIS: Acute situation requiring immediate attention
- TREND: Pattern emerging or accelerating over time
- PIVOT: Actor changing strategy or alignment
- SLEEPER: Quiet development with future significance
- COMPETITION: Two+ actors competing for same objective
- CONSTRAINT: Structural factor limiting options
- LEADER: For the FEATURED story only (special 7-beat structure)
The archetype determines the section's structure (assigned by Structure Agent).
</story_archetypes>

<section_treatments>
- FULL: 300-400 words, complete analysis with futures wheel
- SHORT: 150-200 words, focused on single insight
- COMBINED: Two related clusters in one section (400-500 words total)
- SLEEPER: 100-150 words, plant seed for future coverage
- QUICK_HIT: 1-2 sentences in a bullet list
- KILL: Not included in briefing
</section_treatments>

<featured_selection>
One cluster gets FEATURED treatment (500+ words, prominent placement).
Choose based on:
1. Highest PIC score
2. Strongest "Nub" — which story has the most compelling tension?
3. Most reader interest (major powers, conflict zones)
4. Most surprising/counterintuitive insight
The Featured story always gets archetype "LEADER" (special 7-beat structure).

CRITICAL DISTINCTION:
- FEATURED = A specific cluster story (e.g., "Trump's Diplomatic Agenda")
- HUB (for THEMATIC mode) = A connecting theme/mechanism (e.g., "American Foreign Policy Contradictions")

THESE ARE DIFFERENT THINGS:
- Featured is always A CLUSTER (Trump's Diplomatic Agenda, Iran Crisis, etc.)
- Hub is always A THEME that connects 3+ clusters (NOT a cluster name!)

In THEMATIC mode, the featured cluster is ONE of the spokes, not the hub itself.
</featured_selection>

<narrative_arc>
Every great briefing tells a story. Define the week's through-line:
- What connects the biggest developments?
- What's the "so what" at the global level?

Example arcs:
- "American uncertainty creates openings from Taiwan to Tehran"
- "The emerging axis—China, Russia, Iran—faces its first coordination test"
- "Energy prices are reshaping every alliance calculation"
</narrative_arc>

<section_groups>
Organize sections into 2-4 logical groups with EVOCATIVE titles.

DON'T use: Generic titles like "Regional Developments" or literal cluster lists

DO use titles that tell a story:
- "The Week's Story" — The main event(s) that define this briefing
- "The Pressure Builds" — Situations intensifying
- "Ripple Effects" — Secondary consequences of the main story
- "The Long Game" — Strategic shifts playing out over time
- "On the Radar" — Smaller developments worth watching
- "Sleeper Watch" — Quiet signals that could matter later

For THEMATIC organization:
- "The Hub: American Retrenchment" (the central mechanism)
- "The Spokes" (regional manifestations)

Always provide a group_rationale explaining why sections belong together.
</section_groups>

<transitions>
For EVERY pair of adjacent sections, determine the CONNECTION TYPE, then write the transition.

## TRANSITION DECISION TREE

For each adjacent pair, ask these questions IN ORDER. Stop at the first "yes":

### 1. MECHANISM MANIFESTATION
"Does the same underlying mechanism appear in both, just in different form?"

Example: Economic coercion via tariffs (A) and military coercion via intervention (B)  
Pattern: **"The [mechanism] that appears as [form A] takes [form B] in [context B]..."**

For THEMATIC organization: This should reference your hub mechanism.

### 2. CAUSAL CHAIN
"Does A create conditions that enable, force, or constrain B?"

Example: Venezuela intervention establishes precedent for Taiwan action  
Pattern: **"[A's action] establishes the [precedent/condition] that [B] now faces..."**

### 3. RESOURCE TRADEOFF
"Does attention/bandwidth/capital spent on A reduce capacity for B?"

Example: Diplomatic energy on Greenland reduces Iran focus  
Pattern: **"Every [resource] spent on [A] is [resource] not available for [B]..."**

### 4. CONTRAST
"Do A and B show opposite responses to the same pressure?"

Example: Decisive action in Venezuela vs. hesitation in Ukraine  
Pattern: **"If [A] shows [response X], [B] reveals [opposite response]..."**

### 5. NO CONNECTION
"Are these genuinely unrelated stories?"

If yes: Use **"HARD_BREAK"**

---

## CRITICAL: HARD_BREAK Is Not Failure

A fake connection insults the reader's intelligence. A clean break respects them.
Use HARD_BREAK when none of the 4 connection types fit. This is editorial honesty.

---

## EXAMPLES

❌ **Geographic filler (AVOID):**
"While Washington acts in Latin America, we turn to Asia where China faces challenges..."
Problem: Describes movement, no actual connection

❌ **Temporal filler (AVOID):**
"Meanwhile, in the Middle East, tensions continue..."
Problem: "Meanwhile" is not a connection

✓ **MECHANISM (Connection Type 1):**
"The coercion that appears as tariffs over Greenland takes military form in Caracas, 
where Washington abandoned sanctions for invasion."
Why: Same mechanism (coercion), different tool (economic → military)

✓ **CONSEQUENCE (Connection Type 2):**
"Washington's Venezuela intervention establishes the precedent Beijing has long 
sought for Taiwan: direct action when diplomacy stalls."
Why: A creates condition/precedent enabling B

✓ **TRADEOFF (Connection Type 3):**
"Every diplomatic hour spent managing European anger over Greenland is an hour 
not spent preventing Iran's nuclear acceleration."
Why: Zero-sum resource allocation

✓ **CONTRAST (Connection Type 4):**
"If Venezuela shows what happens when Washington acts decisively, Ukraine reveals 
what happens when it hesitates."
Why: Parallel structure showing opposite outcomes

✓ **HARD_BREAK:**
When stories belong to different section groups with no causal/thematic link, 
just use "HARD_BREAK"—no text needed.

---

## THEMATIC MODE: Hub-Driven Transitions

If you selected THEMATIC organization with a hub mechanism, most transitions should be 
Type 1 (MECHANISM), showing how the hub manifests differently across spokes.

**Example workflow:**
- Hub: "Economic Coercion as Statecraft"
- Spoke A: Greenland tariffs (economic pressure)
- Spoke B: Venezuela intervention (military pressure)
- Connection Type: MECHANISM (same hub, different manifestation)
- Transition: "The coercion that appears as tariffs over Greenland takes military 
  form in Caracas, where Washington abandoned economic sanctions for direct action. 
  Same mechanism, different tool."

This creates structural coherence - every transition reinforces the hub mechanism, 
showing the reader how a single pattern drives multiple regional stories.

---

Format: "SectionA → SectionB": "transition text or HARD_BREAK"

For each transition, STATE THE CONNECTION TYPE first, then write the transition.

Example format in your brief:
```
TRANSITIONS:
- "Greenland → Venezuela"
  TYPE: MECHANISM
  TEXT: "The coercion visible in tariff threats takes military form in Caracas..."

- "Venezuela → Australia"  
  TYPE: HARD_BREAK (no connection)
```
</transitions>

<output_expectations>
Your editorial brief should specify for each cluster:
- Treatment decision (FEATURED, FULL, SHORT, SLEEPER, QUICK_HIT, or KILL)
- If publishing: headline, angle type, archetype, word target, rationale
- If killing: reason (NOTHING_NEW, BELOW_THRESHOLD, REDUNDANT, HOLD_FOR_NEXT_WEEK), PIC score, explanation

Also define:
- Organizing principle (REGIONAL, THEMATIC, or HYBRID)
- If THEMATIC: hub_theme, hub_explanation, hub_angle
- Narrative arc for the week
- Section groupings with evocative titles
- Transitions between sections

SUGGESTED FORMAT (makes it easier for Architect to structure your decisions):
While you can use any format, consider organizing like this:

**FEATURED**: [cluster_name]
- Headline: [your headline]
- Angle: [angle type]
- Archetype: [archetype]
- Word target: [number]
- Rationale: [why featured]

**FULL SECTIONS**: [cluster_1], [cluster_2], [cluster_3]
- [cluster_1]: Headline, angle, archetype, word target, rationale
- [cluster_2]: ...

**SHORT SECTIONS**: [cluster_4], [cluster_5]
- [cluster_4]: Headline, angle, archetype, word target, rationale

**QUICK HITS**: [cluster_6], [cluster_7], [cluster_8]
- [cluster_6]: Headline (1-2 sentences)

**KILLED**: [cluster_9], [cluster_10], ...
- [cluster_9]: Reason, PIC score, explanation
- [cluster_10]: ...

**ORGANIZING**: THEMATIC (or REGIONAL/HYBRID)
- Hub theme: [if THEMATIC]
- Narrative arc: [one sentence]
- Section groups: [evocative titles]
- Transitions: [between sections]

This structure helps the Architect parse your decisions accurately.

Use your thinking budget to reason deeply about each decision.
Use search_web tool to verify significance when unsure about importance.
</output_expectations>

<editorial_memory>
When previous edition context is provided:

1. YOU HAVE MEMORY. Use it.
   - Reference previous coverage naturally
   - "As we noted last week...", "Our forecast about X..."
   - Don't pretend each week is a fresh start

2. AVOID REDUNDANCY
   - Don't re-explain what readers already know
   - Lead with what's NEW, not backstory
   - If same story: 80% new analysis, 20% context

3. TRACK TRAJECTORIES
   - "The situation we flagged last week has now..."
   - "Contrary to our assessment, X happened instead"
   - Note if events validated or contradicted your previous analysis

4. RESPECT HOLDS
   - Stories marked HOLD_FOR_NEXT_WEEK last week deserve fresh consideration
   - They weren't killed—they were deferred for timing reasons
   - Check if the inflection point arrived this week

5. KILL UNCHANGED STORIES
   - If nothing new happened since last week, readers don't need a reminder
   - Exception: Critical strategic context that bears repeating (rare)

6. PREDICTION ACCOUNTABILITY
   - If you made a Sherman Kent forecast last week, acknowledge its resolution
   - Correct predictions build reader trust
   - Incorrect predictions, honestly noted, build more trust
</editorial_memory>
"""


# =============================================================================
# EDITOR AGENT
# =============================================================================

@with_retry(max_attempts=3, initial_delay=2.0, max_delay=30.0)
async def run_editor_agent(
    input_data: EditorInput,
    client: genai.Client | None = None,
) -> EditorDecisions:
    """
    Run the Editor agent for editorial research and decision-making.
    
    Uses deep thinking + Tavily to make informed kill/publish decisions.

    Args:
        input_data: Cluster data and context
        client: Optional Gemini client (created if not provided)

    Returns:
        EditorDecisions with editorial brief and conversation history
    """
    config = get_config()

    if client is None:
        client = genai.Client(api_key=config.gemini_api_key)

    model = config.models.editor
    logger.info(f"   ✏️  Editor reviewing {len(input_data.thematic_clusters)} clusters with {model}...")

    prompt = _build_editor_prompt(input_data)

    # Build generation config - only set params if explicitly configured
    editor_config = types.GenerateContentConfig(
        system_instruction=EDITOR_SYSTEM_PROMPT,
        tools=[get_tavily_search_tool()],
    )
    
    # Only set temperature if explicitly configured
    if config.editor_temperature is not None:
        editor_config.temperature = config.editor_temperature
    
    # Only set max_output_tokens if explicitly configured
    if config.editor_max_output_tokens is not None:
        editor_config.max_output_tokens = config.editor_max_output_tokens
    
    # Only set thinking level if explicitly configured
    if config.editor_thinking_level is not None:
        editor_config.thinking_config = types.ThinkingConfig(
            thinking_level=config.editor_thinking_level,
        )

    # Log config for debugging
    log_model_config("Editor", model, editor_config)

    # Tool loop - allow Editor to research
    messages = [prompt]
    
    for tool_round in range(config.editor_max_tool_rounds):
        response = await client.aio.models.generate_content(
            model=model,
            contents=messages,
            config=editor_config,
        )
        
        # Check for tool calls
        if response.candidates and response.candidates[0].content.parts:
            tool_calls = [
                p for p in response.candidates[0].content.parts 
                if hasattr(p, 'function_call') and p.function_call
            ]
            
            if tool_calls:
                # Execute each tool call
                messages.append(response.candidates[0].content)
                
                tool_results = []
                for part in tool_calls:
                    fc = part.function_call
                    result = execute_tool_call(fc.name, dict(fc.args))
                    tool_results.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"result": result}
                        )
                    )
                
                messages.append(types.Content(role="user", parts=tool_results))
                continue  # Next round
        
        # No more tool calls - editorial decisions complete
        break
    
    # Check why the model stopped
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and "STOP" not in str(finish_reason):
            logger.warning(f"   ⚠️ Editor finish_reason: {finish_reason}")
    
    # Extract editorial brief
    editorial_brief = ""
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                editorial_brief += part.text
    
    if not editorial_brief:
        raise ValueError("Editor produced no output")
    
    # Check if brief seems incomplete (very short or ends mid-sentence)
    if len(editorial_brief) < 2000:
        logger.warning(f"   ⚠️ Editor brief seems short: {len(editorial_brief)} chars")
    
    logger.info(f"   ✅ Editor complete ({len(messages)} messages, {len(editorial_brief)} chars)")
    
    return EditorDecisions(
        editorial_brief=editorial_brief,
        conversation_history=messages,
    )
