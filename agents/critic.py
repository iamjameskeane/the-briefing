"""
Critic Agent - Split Content/Style Evaluation

This module provides two critic functions:
1. ContentCritic - Evaluates factual/analytical quality (for Writer loop)
2. StyleCritic - Evaluates prose quality (for Stylist loop)

Architecture:
    Analyst Output → Writer → ContentCritic → (retry loop)
                         ↓
                  Approved Writer Draft
                         ↓
                    Stylist → StyleCritic → (retry loop)
                         ↓
                    Final Output

Key Design Principles:
- Writer's source of truth = Analyst output
- Stylist's source of truth = Writer's approved draft
- Content issues never bounce to Stylist
- Style issues never bounce to Writer
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from google import genai

from state import (
    ContentCriticResult,
    StyleCriticResult,
    CoVeResult,
    FactualClaim,
    VerificationResult,
    HallucinationFlag,
    SacredElements,
)
from tools import get_search_results_for_verification
from utils import logger
from .schemas import AnalystOutput

if TYPE_CHECKING:
    from state import SectionBlueprint


# =============================================================================
# BANNED PHRASES AND CONSTANTS
# =============================================================================

BANNED_PHRASES = [
    "remains to be seen",
    "time will tell",
    "could potentially",
    "significant developments",
    "various factors",
    "the situation is evolving",
    "in the coming weeks",
    "major implications",
    "it is unclear",
    "only time will tell",
    "the jury is still out",
    "at the end of the day",
    "moving forward",
    "perfect storm",
    "game changer",
]

SHERMAN_KENT_TERMS = [
    "ALMOST CERTAIN",
    "HIGHLY LIKELY",
    "LIKELY",
    "ROUGHLY EVEN",
    "UNLIKELY",
    "HIGHLY UNLIKELY",
    "REMOTE",
]

# Pass thresholds
CONTENT_PASS_THRESHOLD = 75  # Out of 100
STYLE_PASS_THRESHOLD = 70    # Out of 100


# =============================================================================
# CONTENT CRITIC (for Writer Loop)
# =============================================================================

def _check_angle_delivery(draft: str, editorial_angle: str) -> tuple[int, list[str]]:
    """
    Check if the draft actually delivers the specified editorial angle.
    
    Returns:
        (score out of 15, issues list)
    """
    issues = []
    
    # Get first paragraph
    paragraphs = draft.split("\n\n")
    first_para = paragraphs[0] if paragraphs else draft
    first_para_lower = first_para.lower()
    
    angle_upper = editorial_angle.upper()
    
    # MISSING_THE_POINT: Should have "everyone is..." or "but/yet/the real story"
    if "MISSING_THE_POINT" in angle_upper or "MISSING THE POINT" in angle_upper:
        indicators = [
            "everyone is", "everyone's", "conventional", "most coverage",
            "but the real", "yet the real", "the real story", "what's missing",
            "overlooked", "ignored"
        ]
        if any(ind in first_para_lower for ind in indicators):
            return 15, []
        issues.append(f"Angle not delivered: Opening should execute '{editorial_angle[:50]}...'")
        return 5, issues
    
    # UNINTENDED_CONSEQUENCE: Should mention intended goal vs actual result
    if "UNINTENDED_CONSEQUENCE" in angle_upper or "UNINTENDED CONSEQUENCE" in angle_upper:
        indicators = [
            "designed to", "intended to", "meant to", "instead",
            "backfired", "ironically", "opposite", "unintended"
        ]
        if any(ind in first_para_lower for ind in indicators):
            return 15, []
        issues.append(f"Angle not delivered: Show intended vs actual result")
        return 5, issues
    
    # CONSTRAINT_REVEALED: Should explicitly state constraint
    if "CONSTRAINT_REVEALED" in angle_upper or "CONSTRAINT REVEALED" in angle_upper:
        indicators = [
            "exposed what", "revealed", "always true", "structural",
            "cannot", "unable to", "lacks", "without"
        ]
        if sum(1 for ind in indicators if ind in first_para_lower) >= 2:
            return 15, []
        issues.append(f"Angle not delivered: Expose the structural constraint")
        return 5, issues
    
    # TRAJECTORY_SHIFT: Should mark before/after
    if "TRAJECTORY_SHIFT" in angle_upper or "TRAJECTORY SHIFT" in angle_upper:
        indicators = [
            "after years", "after months", "pivoted", "shift", "changed",
            "reversed", "abandoned", "now", "turned"
        ]
        if any(ind in first_para_lower for ind in indicators):
            return 15, []
        issues.append(f"Angle not delivered: Mark the before/after trajectory shift")
        return 5, issues
    
    # MIND_CLEARING: Should have "forget X, this is about Y"
    if "MIND_CLEARING" in angle_upper or "MIND CLEARING" in angle_upper:
        indicators = [
            "forget", "ignore", "this is about", "this is really about",
            "the key is", "what matters is", "boils down to"
        ]
        if any(ind in first_para_lower for ind in indicators):
            return 15, []
        issues.append(f"Angle not delivered: Use 'forget X, this is about Y' framing")
        return 5, issues
    
    # For other/custom angles, just check if it appears somewhat near the top
    return 15, []  # Give benefit of doubt for unknown angles


def _check_predictions(draft: str) -> tuple[int, list[str]]:
    """
    Check if the draft makes predictions (the "So What?").
    
    Returns:
        (score out of 15, issues list)
    """
    draft_lower = draft.lower()
    
    prediction_indicators = [
        "will ", "expect", "likely", "forecast", "anticipate",
        "predict", "trajectory", "heading toward", "set to"
    ]
    
    has_prediction = any(ind in draft_lower for ind in prediction_indicators)
    
    if has_prediction:
        return 15, []
    return 0, ["Missing prediction: Add forward-looking assessment with probability term"]


def _check_constraints(draft: str) -> tuple[int, list[str]]:
    """
    Check if actor constraints are explicitly stated.
    
    Returns:
        (score out of 15, issues list)
    """
    draft_lower = draft.lower()
    
    constraint_indicators = [
        "constrained by", "limited by", "forced to", "cannot",
        "prevents", "unable to", "lacks the", "without the"
    ]
    
    constraints_found = sum(1 for ind in constraint_indicators if ind in draft_lower)
    
    if constraints_found >= 2:
        return 15, []
    elif constraints_found == 1:
        return 8, ["Add more explicit constraints: What can actors NOT do?"]
    return 0, ["Missing constraints: State what limits actor options"]


def _check_entity_density(draft: str) -> tuple[int, list[str]]:
    """
    Check entity density in first paragraph.
    
    Returns:
        (score out of 10, issues list)
    """
    paragraphs = draft.split("\n\n")
    first_para = paragraphs[0] if paragraphs else draft
    words = first_para.split()
    
    # Count capitalized words (likely proper nouns)
    capitalized = [w for w in words if w and w[0].isupper() and len(w) > 2]
    
    # Count numbers
    numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?%?", first_para)
    
    entity_count = len(capitalized) + len(numbers)
    
    if entity_count >= 5:
        return 10, []
    elif entity_count >= 3:
        return 5, [f"Low density: Only {entity_count} entities in first paragraph (need 5+)"]
    return 0, [f"Very low density: Only {entity_count} entities. Add specific names, numbers, dates"]


def _check_sherman_kent(draft: str) -> tuple[int, list[str]]:
    """
    Check for Sherman Kent probability terms.
    
    Returns:
        (score out of 10, issues list)
    """
    kent_found = any(term in draft.upper() for term in SHERMAN_KENT_TERMS)
    
    if kent_found:
        return 10, []
    return 0, ["Missing Sherman Kent probability term (e.g., 'HIGHLY LIKELY')"]


def _check_coverage(draft: str, analyst_output: AnalystOutput) -> tuple[int, list[str]]:
    """
    Check if writer used analyst's key insights.
    
    Returns:
        (score out of 10, issues list)
    """
    draft_lower = draft.lower()
    
    # Check third-order effect
    third_order = analyst_output.futures_wheel.third_order.lower()
    has_third_order = any(
        word in draft_lower
        for word in third_order.split()[:5]
        if len(word) > 4
    )
    
    if has_third_order:
        return 10, []
    return 0, ["Missing third-order effect from analysis"]


def _check_sacred_elements(draft: str, sacred_elements: SacredElements) -> tuple[int, list[str]]:
    """
    Check if sacred elements are preserved.
    
    Returns:
        (score out of 10, issues list)
    """
    issues = []
    score = 10
    
    # Check proper nouns
    for noun in sacred_elements.proper_nouns[:5]:
        if noun not in draft:
            issues.append(f"Missing proper noun: {noun}")
            score -= 2
    
    # Check statistics
    for stat in sacred_elements.statistics[:3]:
        if stat not in draft:
            issues.append(f"Missing statistic: {stat}")
            score -= 2
    
    return max(0, score), issues


# =============================================================================
# CHAIN OF VERIFICATION (CoVe)
# =============================================================================

def extract_factual_claims(draft: str) -> list[FactualClaim]:
    """
    Extract factual claims from draft for verification.
    
    Looks for:
    - Statistics (numbers, percentages) - excluding URLs and probability ranges
    - Dates (but NOT dates in the Sources section)
    - Named entities with actions
    - Quotes
    
    IMPORTANT: Excludes the **Sources:** section to avoid false positives.
    """
    claims = []
    claim_id = 0
    
    # Strip the Sources section
    draft_to_check = draft
    sources_patterns = [
        r'\*\*Sources:\*\*.*$',
        r'## Sources.*$',
        r'\nSources:.*$',
    ]
    for pattern in sources_patterns:
        draft_to_check = re.sub(pattern, '', draft_to_check, flags=re.DOTALL | re.IGNORECASE)
    
    sentences = re.split(r'(?<!\d)[.!?]+(?!\d)', draft_to_check)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    for sentence in sentences:
        # Skip sentences with URLs
        if 'http://' in sentence or 'https://' in sentence or 'event=' in sentence:
            continue
        
        # Remove probability expressions before extracting stats
        sentence_cleaned = re.sub(
            r'\b(?:HIGHLY LIKELY|LIKELY|POSSIBLE|PROBABLE|UNLIKELY)\s*\(\d+-\d+%?\)',
            '',
            sentence,
            flags=re.IGNORECASE
        )
        
        # Find statistics
        stats = re.findall(
            r'(?<!\S)\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|thousand|percent|%))?(?=\s|$|[,.\)\]])',
            sentence_cleaned,
            re.IGNORECASE
        )
        
        for stat in stats:
            stat_clean = stat.strip()
            has_suffix = any(s in stat_clean.lower() for s in ['%', 'billion', 'million', 'trillion', 'thousand', 'percent'])
            if len(stat_clean) <= 2 and not has_suffix:
                continue
            if re.match(r'^(19|20)\d{2}$', stat_clean):
                continue
            if stat_clean.endswith(','):
                continue
            
            claims.append(FactualClaim(
                id=f"claim_{claim_id}",
                claim_text=stat_clean,
                claim_type="STATISTIC",
                source_sentence=sentence
            ))
            claim_id += 1
        
        # Find dates
        if not re.search(r'\[\d+\]', sentence):
            dates = re.findall(
                r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?',
                sentence,
                re.IGNORECASE
            )
            for date in dates:
                claims.append(FactualClaim(
                    id=f"claim_{claim_id}",
                    claim_text=date,
                    claim_type="DATE",
                    source_sentence=sentence
                ))
                claim_id += 1
    
    return claims[:20]


def verify_claims_against_sources(
    claims: list[FactualClaim],
    source_events: list[dict],
    external_sources: list[dict] | None = None,
    search_results_text: str = "",
) -> tuple[list[VerificationResult], list[HallucinationFlag]]:
    """
    Verify claims against ALL source documents.
    """
    results = []
    flags = []
    
    # Build source text
    source_text = ""
    for event in source_events:
        source_text += f" {event.get('title', '')} {event.get('summary', '')} "
    
    if external_sources:
        for src in external_sources:
            source_text += f" {src.get('title', '')} {src.get('content', '')} "
    
    if search_results_text:
        source_text += f" {search_results_text} "
    
    source_text = source_text.lower()
    
    for claim in claims:
        claim_text_lower = claim.claim_text.lower()
        match = claim_text_lower in source_text
        
        if claim.claim_type == "STATISTIC":
            numbers_in_claim = re.findall(r'[\d,]+(?:\.\d+)?', claim.claim_text)
            match = any(num in source_text for num in numbers_in_claim)
        
        results.append(VerificationResult(
            claim_id=claim.id,
            verification_question=f"Does '{claim.claim_text}' appear in sources?",
            independent_answer="Found in sources" if match else "Not found in sources",
            draft_answer=claim.claim_text,
            match=match,
            confidence="MEDIUM" if match else "LOW"
        ))
        
        if not match:
            severity = "CRITICAL" if claim.claim_type == "STATISTIC" else "WARNING"
            flags.append(HallucinationFlag(
                claim_id=claim.id,
                severity=severity,
                explanation=f"Could not verify '{claim.claim_text}' in source documents"
            ))
    
    return results, flags


def run_cove(
    draft: str,
    source_events: list[dict],
    external_sources: list[dict] | None = None,
    search_results_text: str = "",
) -> CoVeResult:
    """Run Chain of Verification on the draft."""
    claims = extract_factual_claims(draft)
    
    if not claims:
        return CoVeResult(
            claims_extracted=[],
            verification_results=[],
            hallucination_flags=[]
        )
    
    results, flags = verify_claims_against_sources(
        claims, 
        source_events,
        external_sources=external_sources,
        search_results_text=search_results_text,
    )
    
    return CoVeResult(
        claims_extracted=claims,
        verification_results=results,
        hallucination_flags=flags
    )


# =============================================================================
# CONTENT CRITIC MAIN FUNCTION
# =============================================================================

def run_content_critic(
    draft: str,
    analyst_output: AnalystOutput,
    sacred_elements: SacredElements,
    source_events: list[dict],
    editorial_angle: Optional[str] = None,
    region: str = "",
) -> ContentCriticResult:
    """
    Run the Content Critic on a Writer draft.
    
    Evaluates factual/analytical quality:
    - Angle delivery
    - Predictions (So What?)
    - Constraints
    - Entity density
    - Sherman Kent terms
    - Coverage of analyst insights
    - Sacred elements preservation
    - CoVe hallucination detection
    
    Args:
        draft: The Writer's draft
        analyst_output: The source analysis
        sacred_elements: Facts that must be preserved
        source_events: Source documents for CoVe
        editorial_angle: Optional angle to check delivery
        region: Region name for logging
        
    Returns:
        ContentCriticResult with score, issues, and feedback
    """
    logger.info(f"   🔍 [{region}] Running Content Critic...")
    
    all_issues = []
    
    # 1. Angle Delivery (15 points)
    if editorial_angle:
        angle_score, angle_issues = _check_angle_delivery(draft, editorial_angle)
    else:
        angle_score = 15  # No angle specified, give full points
        angle_issues = []
    all_issues.extend(angle_issues)
    
    # 2. Predictions (15 points)
    prediction_score, prediction_issues = _check_predictions(draft)
    all_issues.extend(prediction_issues)
    
    # 3. Constraints (15 points)
    constraints_score, constraints_issues = _check_constraints(draft)
    all_issues.extend(constraints_issues)
    
    # 4. Entity Density (10 points)
    density_score, density_issues = _check_entity_density(draft)
    all_issues.extend(density_issues)
    
    # 5. Sherman Kent (10 points)
    kent_score, kent_issues = _check_sherman_kent(draft)
    all_issues.extend(kent_issues)
    
    # 6. Coverage (10 points)
    coverage_score, coverage_issues = _check_coverage(draft, analyst_output)
    all_issues.extend(coverage_issues)
    
    # 7. Sacred Elements (10 points)
    sacred_score, sacred_issues = _check_sacred_elements(draft, sacred_elements)
    all_issues.extend(sacred_issues)
    
    # 8. CoVe (15 points) - deduct for hallucinations
    external_sources = getattr(analyst_output, 'external_sources', None)
    search_results_text = get_search_results_for_verification()
    
    cove_result = run_cove(
        draft,
        source_events,
        external_sources=external_sources,
        search_results_text=search_results_text,
    )
    
    cove_score = 15
    if cove_result.has_critical_hallucinations:
        cove_score = 0
        all_issues.insert(0, "CRITICAL: Unverified statistics detected. Check source documents.")
    elif cove_result.hallucination_flags:
        cove_score = max(0, 15 - len(cove_result.hallucination_flags) * 3)
    
    # Calculate total
    total_score = (
        angle_score + 
        prediction_score + 
        constraints_score + 
        density_score + 
        kent_score + 
        coverage_score + 
        sacred_score +
        cove_score
    )
    
    passed = total_score >= CONTENT_PASS_THRESHOLD
    
    # Build feedback
    feedback = ""
    if all_issues:
        feedback = "CONTENT ISSUES TO FIX:\n" + "\n".join(f"- {issue}" for issue in all_issues)
        feedback += f"\n\nYour score: {total_score}/100 (need {CONTENT_PASS_THRESHOLD} to pass)"
    
    # Log result
    status = "✅ PASSED" if passed else "❌ FAILED"
    logger.info(f"   {status} [{region}] Content Score: {total_score}/100")
    
    if cove_result.hallucination_flags:
        logger.warning(f"   ⚠️ [{region}] Hallucination flags: {len(cove_result.hallucination_flags)}")
    
    return ContentCriticResult(
        passed=passed,
        score=total_score,
        angle_delivery_score=angle_score,
        prediction_score=prediction_score,
        constraints_score=constraints_score,
        density_score=density_score,
        sherman_kent_score=kent_score,
        coverage_score=coverage_score,
        sacred_elements_score=sacred_score,
        cove_result=cove_result,
        issues=all_issues,
        feedback=feedback,
    )


# =============================================================================
# STYLE CRITIC (for Stylist Loop)
# =============================================================================

def _check_weasel_free(draft: str) -> tuple[int, list[str]]:
    """
    Check for banned phrases.
    
    Returns:
        (score out of 20, issues list)
    """
    draft_lower = draft.lower()
    weasels_found = [p for p in BANNED_PHRASES if p.lower() in draft_lower]
    
    if not weasels_found:
        return 20, []
    
    deduction = min(20, len(weasels_found) * 5)
    return 20 - deduction, [f"Remove banned phrases: {weasels_found[:3]}"]


def _check_sentence_variety(draft: str) -> tuple[int, list[str]]:
    """
    Check for varied sentence openings.
    
    Returns:
        (score out of 15, issues list)
    """
    sentences = re.split(r'[.!?]+', draft)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 3:
        return 10, []  # Not enough to evaluate
    
    starters = [s.split()[0].lower() if s.split() else "" for s in sentences]
    
    # Count consecutive same starters
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(starters)):
        if starters[i] == starters[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    
    if max_consecutive >= 3:
        return 5, [f"Repetitive starts: {max_consecutive} consecutive sentences start the same way"]
    elif max_consecutive == 2:
        return 12, ["Slight repetition: Some consecutive sentences start the same"]
    return 15, []


def _check_active_voice(draft: str) -> tuple[int, list[str]]:
    """
    Check for passive voice usage.
    
    Returns:
        (score out of 15, issues list)
    """
    draft_lower = draft.lower()
    
    passive_indicators = [
        " was ", " were ", " been ", " being ",
        " is being ", " was being ", " have been ", " has been "
    ]
    
    passive_count = sum(1 for ind in passive_indicators if ind in draft_lower)
    
    if passive_count <= 2:
        return 15, []
    elif passive_count <= 4:
        return 10, [f"Reduce passive voice: Found {passive_count} instances"]
    return 5, [f"Too much passive voice: {passive_count} instances. Use active verbs"]


def _check_opening_punch(draft: str) -> tuple[int, list[str]]:
    """
    Check for strong opening.
    
    Returns:
        (score out of 15, issues list)
    """
    sentences = re.split(r'[.!?]+', draft)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0, ["No opening sentence found"]
    
    first_sentence = sentences[0]
    first_words = first_sentence.split()[:5]
    
    weak_openers = ["the", "there", "it", "this", "in", "on", "a", "an"]
    
    if first_words and first_words[0].lower() not in weak_openers:
        return 15, []
    elif first_words and len(first_sentence.split()) < 10:  # Short punchy opener
        return 15, []
    
    return 8, ["Strengthen opening: Don't start with 'The' or weak constructions"]


def _check_closing_resonance(draft: str) -> tuple[int, list[str]]:
    """
    Check for strong closing.
    
    Returns:
        (score out of 10, issues list)
    """
    sentences = re.split(r'[.!?]+', draft)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0, ["No closing sentence found"]
    
    last_sentence = sentences[-1]
    
    closing_indicators = ["will", "expect", "?", "watch", "trigger", "if", "when"]
    has_strong_close = any(ind in last_sentence.lower() for ind in closing_indicators)
    
    if has_strong_close:
        return 10, []
    return 5, ["Strengthen closing: End with prediction, warning, or memorable line"]


def _check_quotable_line(draft: str) -> tuple[int, list[str]]:
    """
    Check for one memorable/quotable phrase.
    
    Returns:
        (score out of 10, issues list)
    """
    sentences = re.split(r'[.!?]+', draft)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Short punchy sentences (3-8 words)
    short_punchy = any(3 <= len(s.split()) <= 8 for s in sentences)
    
    # Memorable construction (em dash or colon)
    has_construction = " — " in draft or ": " in draft
    
    if short_punchy and has_construction:
        return 10, []
    elif short_punchy or has_construction:
        return 7, ["Add one memorable line with punchy construction (dash, colon)"]
    return 3, ["Missing memorable line: Add one quotable phrase readers will remember"]


def _check_orwell_compliance(draft: str) -> tuple[int, list[str]]:
    """
    Check for Orwell's rules compliance (short words, no jargon).
    
    Returns:
        (score out of 15, issues list)
    """
    issues = []
    score = 15
    
    # Check for overly long words (potential jargon)
    words = draft.split()
    long_words = [w for w in words if len(w) > 12 and w.isalpha()]
    
    if len(long_words) > 5:
        score -= 5
        issues.append(f"Too many long words: {long_words[:3]}... Use shorter alternatives")
    
    # Check for common jargon
    jargon = ["utilize", "leverage", "synergy", "paradigm", "holistic", "proactive"]
    jargon_found = [j for j in jargon if j in draft.lower()]
    
    if jargon_found:
        score -= len(jargon_found) * 3
        issues.append(f"Remove jargon: {jargon_found}")
    
    return max(0, score), issues


def _check_fact_preservation(styled_draft: str, writer_draft: str, sacred_elements: SacredElements) -> tuple[int, list[str]]:
    """
    Check that Stylist preserved Writer's facts.
    
    Returns:
        (score out of 0 - this is a pass/fail check that can deduct)
    """
    issues = []
    
    # Check sacred elements are still present
    for noun in sacred_elements.proper_nouns[:5]:
        if noun in writer_draft and noun not in styled_draft:
            issues.append(f"Stylist removed proper noun: {noun}")
    
    for stat in sacred_elements.statistics[:5]:
        if stat in writer_draft and stat not in styled_draft:
            issues.append(f"Stylist removed statistic: {stat}")
    
    for date in sacred_elements.dates[:3]:
        if date in writer_draft and date not in styled_draft:
            issues.append(f"Stylist removed date: {date}")
    
    # This check doesn't add points, but issues cause failure
    return 0, issues


# =============================================================================
# CONTENT PRESERVATION CHECKS (run on styled draft)
# =============================================================================

def _check_content_preserved(
    styled_draft: str,
    writer_draft: str,
) -> tuple[int, list[str]]:
    """
    Check that Stylist preserved the objective content from Writer.
    
    Compares key content indicators between writer and styled drafts.
    Stylist should NOT reduce:
    - Predictions (forward-looking language)
    - Constraints (actor limitations)
    - Entity density
    - Sherman Kent terms
    
    Returns:
        (penalty score to deduct, issues list)
    """
    issues = []
    penalty = 0
    
    styled_lower = styled_draft.lower()
    writer_lower = writer_draft.lower()
    
    # 1. Check predictions preserved
    prediction_indicators = [
        "will ", "expect", "likely", "forecast", "anticipate",
        "predict", "trajectory", "heading toward", "set to"
    ]
    writer_predictions = sum(1 for ind in prediction_indicators if ind in writer_lower)
    styled_predictions = sum(1 for ind in prediction_indicators if ind in styled_lower)
    
    if styled_predictions < writer_predictions:
        lost = writer_predictions - styled_predictions
        issues.append(f"Lost {lost} prediction indicator(s) - restore forward-looking language")
        penalty += lost * 5
    
    # 2. Check constraints preserved
    constraint_indicators = [
        "constrained by", "limited by", "forced to", "cannot",
        "prevents", "unable to", "lacks the", "without the"
    ]
    writer_constraints = sum(1 for ind in constraint_indicators if ind in writer_lower)
    styled_constraints = sum(1 for ind in constraint_indicators if ind in styled_lower)
    
    if styled_constraints < writer_constraints:
        lost = writer_constraints - styled_constraints
        issues.append(f"Lost {lost} constraint phrase(s) - restore actor limitations")
        penalty += lost * 5
    
    # 3. Check Sherman Kent terms preserved
    writer_kent = sum(1 for term in SHERMAN_KENT_TERMS if term in writer_draft.upper())
    styled_kent = sum(1 for term in SHERMAN_KENT_TERMS if term in styled_draft.upper())
    
    if styled_kent < writer_kent:
        issues.append("Lost Sherman Kent probability term(s) - restore HIGHLY LIKELY, LIKELY, etc.")
        penalty += 15  # Critical - this is a major loss
    
    # 4. Check entity density preserved (first paragraph)
    def count_entities(text: str) -> int:
        paragraphs = text.split("\n\n")
        first_para = paragraphs[0] if paragraphs else text
        words = first_para.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 2]
        numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?%?", first_para)
        return len(capitalized) + len(numbers)
    
    writer_entities = count_entities(writer_draft)
    styled_entities = count_entities(styled_draft)
    
    if styled_entities < writer_entities - 2:  # Allow small variance
        lost = writer_entities - styled_entities
        issues.append(f"Reduced entity density by {lost} - restore specific names/numbers")
        penalty += lost * 2
    
    # 5. Check that key specific facts weren't genericized
    # Look for numbers in writer that disappeared in styled
    writer_numbers = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?%?', writer_draft))
    styled_numbers = set(re.findall(r'\d+(?:,\d+)*(?:\.\d+)?%?', styled_draft))
    
    lost_numbers = writer_numbers - styled_numbers
    # Filter out trivial numbers (1, 2, etc.)
    significant_lost = [n for n in lost_numbers if len(n) > 2 or '%' in n]
    
    if significant_lost:
        issues.append(f"Lost specific numbers: {list(significant_lost)[:3]} - don't replace data with prose")
        penalty += len(significant_lost) * 3
    
    return min(penalty, 50), issues  # Cap penalty at 50


# =============================================================================
# STYLE CRITIC MAIN FUNCTION
# =============================================================================

def run_style_critic(
    styled_draft: str,
    writer_draft: str,
    sacred_elements: SacredElements,
    region: str = "",
) -> StyleCriticResult:
    """
    Run the Style Critic on a Stylist draft.
    
    Evaluates BOTH prose quality AND content preservation:
    
    Prose Quality:
    - Weasel-free (no banned phrases)
    - Sentence variety
    - Active voice
    - Opening punch
    - Closing resonance
    - Quotable line
    - Orwell compliance
    
    Content Preservation (Stylist must not break Writer's work):
    - Sacred elements (proper nouns, statistics, dates)
    - Predictions and forward-looking language
    - Constraints and actor limitations
    - Sherman Kent probability terms
    - Entity density
    
    Args:
        styled_draft: The Stylist's output
        writer_draft: The Writer's original (source of truth for facts AND analysis)
        sacred_elements: Facts that must be preserved
        region: Region name for logging
        
    Returns:
        StyleCriticResult with score, issues, and feedback
    """
    logger.info(f"   🎨 [{region}] Running Style Critic...")
    
    all_issues = []
    
    # =========================================================================
    # PROSE QUALITY CHECKS (what Stylist should IMPROVE)
    # =========================================================================
    
    # 1. Weasel-free (20 points)
    weasel_score, weasel_issues = _check_weasel_free(styled_draft)
    all_issues.extend(weasel_issues)
    
    # 2. Sentence Variety (15 points)
    variety_score, variety_issues = _check_sentence_variety(styled_draft)
    all_issues.extend(variety_issues)
    
    # 3. Active Voice (15 points)
    active_score, active_issues = _check_active_voice(styled_draft)
    all_issues.extend(active_issues)
    
    # 4. Opening Punch (15 points)
    opening_score, opening_issues = _check_opening_punch(styled_draft)
    all_issues.extend(opening_issues)
    
    # 5. Closing Resonance (10 points)
    closing_score, closing_issues = _check_closing_resonance(styled_draft)
    all_issues.extend(closing_issues)
    
    # 6. Quotable Line (10 points)
    quotable_score, quotable_issues = _check_quotable_line(styled_draft)
    all_issues.extend(quotable_issues)
    
    # 7. Orwell Compliance (15 points)
    orwell_score, orwell_issues = _check_orwell_compliance(styled_draft)
    all_issues.extend(orwell_issues)
    
    # Calculate base style score
    style_score = (
        weasel_score +
        variety_score +
        active_score +
        opening_score +
        closing_score +
        quotable_score +
        orwell_score
    )
    
    # =========================================================================
    # CONTENT PRESERVATION CHECKS (what Stylist must NOT break)
    # =========================================================================
    
    # 8. Sacred Elements (hard failure if broken)
    _, preservation_issues = _check_fact_preservation(styled_draft, writer_draft, sacred_elements)
    
    # 9. Content Quality Preserved (penalty for regression)
    content_penalty, content_issues = _check_content_preserved(styled_draft, writer_draft)
    
    # Add content issues with clear labeling
    if content_issues:
        all_issues.insert(0, "⚠️ CONTENT REGRESSION (Stylist broke Writer's analysis):")
        all_issues.extend([f"  {issue}" for issue in content_issues])
    
    if preservation_issues:
        all_issues.insert(0, "🚨 SACRED ELEMENTS ALTERED:")
        all_issues.extend([f"  {issue}" for issue in preservation_issues])
    
    # Calculate final score (style score minus content regression penalty)
    total_score = max(0, style_score - content_penalty)
    
    # Determine pass/fail
    # Fail if: sacred elements broken, OR score too low, OR severe content regression
    critical_content_failure = content_penalty >= 30  # Lost too much analytical content
    
    if preservation_issues:
        passed = False
        all_issues.insert(0, "CRITICAL: Stylist altered sacred elements")
    elif critical_content_failure:
        passed = False
        all_issues.insert(0, f"CRITICAL: Stylist degraded content quality (penalty: -{content_penalty})")
    else:
        passed = total_score >= STYLE_PASS_THRESHOLD
    
    # Build feedback
    feedback = ""
    if all_issues:
        feedback = "STYLE ISSUES TO FIX:\n" + "\n".join(f"- {issue}" for issue in all_issues)
        feedback += f"\n\nYour score: {total_score}/100 (need {STYLE_PASS_THRESHOLD} to pass)"
        if content_penalty > 0:
            feedback += f"\nContent regression penalty: -{content_penalty}"
        feedback += "\n\nREMINDER: Style means HOW you say it, not WHAT you say."
        feedback += "\nDo NOT remove predictions, constraints, numbers, or Sherman Kent terms."
    
    # Log result
    status = "✅ PASSED" if passed else "❌ FAILED"
    logger.info(f"   {status} [{region}] Style Score: {total_score}/100 (penalty: -{content_penalty})")
    
    if content_penalty > 0:
        logger.warning(f"   ⚠️ [{region}] Content regression detected: -{content_penalty} points")
    
    return StyleCriticResult(
        passed=passed,
        score=total_score,
        weasel_free_score=weasel_score,
        sentence_variety_score=variety_score,
        active_voice_score=active_score,
        opening_punch_score=opening_score,
        closing_resonance_score=closing_score,
        quotable_line_score=quotable_score,
        orwell_score=orwell_score,
        fact_preservation_score=0 if (preservation_issues or content_penalty >= 30) else 10,
        issues=all_issues,
        feedback=feedback,
    )


# =============================================================================
# WRITER LOOP
# =============================================================================

async def run_writer_loop(
    analyst_output: AnalystOutput,
    events: list[dict],
    blueprint: "SectionBlueprint",
    sacred_elements: SacredElements,
    editorial_angle: Optional[str] = None,
    max_attempts: int = 3,
    client: genai.Client | None = None,
) -> tuple[str, ContentCriticResult]:
    """
    Run Writer with ContentCritic feedback loop.
    
    The Writer always has access to:
    - analyst_output (source of truth)
    - previous_draft (if retry)
    - content_feedback (if retry)
    
    Returns:
        (approved_draft, final_critic_result)
    """
    from .writer import run_writer_agent
    from .schemas import WriterInput
    
    region = analyst_output.region
    previous_draft = None
    content_feedback = None
    
    for attempt in range(max_attempts):
        logger.info(f"   📝 [{region}] Writer attempt {attempt + 1}/{max_attempts}...")
        
        # Build input with full context
        writer_input = WriterInput(
            analyst_output=analyst_output,
            events_for_linking=events,
            section_type=blueprint.archetype.lower() if blueprint else "regional",
            word_limit=blueprint.word_target if blueprint else 300,
            section_blueprint=blueprint,
            sacred_elements=sacred_elements,
            editorial_angle=editorial_angle,
            # Revision context
            previous_draft=previous_draft,
            revision_feedback=content_feedback,
            apply_chain_of_density=True,
        )
        
        # Generate draft
        draft = await run_writer_agent(writer_input, client)
        
        # Critique content only
        result = run_content_critic(
            draft=draft,
            analyst_output=analyst_output,
            sacred_elements=sacred_elements,
            source_events=events,
            editorial_angle=editorial_angle,
            region=region,
        )
        
        if result.passed:
            logger.info(f"   ✅ [{region}] Writer approved on attempt {attempt + 1}: {result.score}/100")
            return draft, result
        
        # Prepare for retry
        previous_draft = draft
        content_feedback = result.feedback
        
        logger.info(f"   🔄 [{region}] Writer attempt {attempt + 1} failed ({result.score}/100), retrying...")
    
    # Max attempts reached - return best effort
    logger.warning(f"   ⚠️ [{region}] Writer max attempts reached, accepting with score {result.score}")
    return draft, result


# =============================================================================
# STYLIST LOOP
# =============================================================================

async def run_stylist_loop(
    writer_draft: str,
    blueprint: "SectionBlueprint",
    sacred_elements: SacredElements,
    max_attempts: int = 3,
    client: genai.Client | None = None,
) -> tuple[str, StyleCriticResult]:
    """
    Run Stylist with StyleCritic feedback loop.
    
    The writer_draft is the source of truth for facts.
    Stylist can only change HOW things are said, not WHAT is said.
    
    The Stylist always has access to:
    - writer_draft (source of truth - never changes)
    - previous_styled (if retry)
    - style_feedback (if retry)
    
    Returns:
        (styled_draft, final_critic_result)
    """
    from .stylist import run_stylist_agent, StylistInput
    
    region = blueprint.region
    previous_styled = None
    style_feedback = None
    
    for attempt in range(max_attempts):
        logger.info(f"   ✨ [{region}] Stylist attempt {attempt + 1}/{max_attempts}...")
        
        # Build input with full context
        stylist_input = StylistInput(
            region=region,
            writer_draft=writer_draft,  # Always the approved writer draft
            blueprint=blueprint,
            sacred_elements=sacred_elements,
            # Revision context
            previous_styled=previous_styled,
            critic_feedback=style_feedback,
        )
        
        # Transform prose
        styled = await run_stylist_agent(stylist_input, client)
        
        # Critique style only
        result = run_style_critic(
            styled_draft=styled,
            writer_draft=writer_draft,  # For fact preservation check
            sacred_elements=sacred_elements,
            region=region,
        )
        
        if result.passed:
            logger.info(f"   ✅ [{region}] Stylist approved on attempt {attempt + 1}: {result.score}/100")
            return styled, result
        
        # Prepare for retry
        previous_styled = styled
        style_feedback = result.feedback
        
        logger.info(f"   🔄 [{region}] Stylist attempt {attempt + 1} failed ({result.score}/100), retrying...")
    
    # Max attempts reached
    logger.warning(f"   ⚠️ [{region}] Stylist max attempts reached, accepting with score {result.score}")
    return styled, result


# =============================================================================
# COMBINED CONTENT PIPELINE
# =============================================================================

async def run_content_pipeline(
    analyst_output: AnalystOutput,
    events: list[dict],
    blueprint: "SectionBlueprint",
    sacred_elements: SacredElements,
    editorial_angle: Optional[str] = None,
    max_writer_attempts: int = 3,
    max_stylist_attempts: int = 3,
    client: genai.Client | None = None,
) -> tuple[str, str, ContentCriticResult, StyleCriticResult]:
    """
    Run the full Writer → Stylist pipeline with independent critic loops.
    
    Architecture:
        Analyst Output → Writer Loop → Approved Draft → Stylist Loop → Final
    
    Returns:
        (writer_draft, styled_draft, content_result, style_result)
    """
    region = analyst_output.region
    
    # Phase 1: Writer Loop
    logger.info(f"   ═══ [{region}] Starting Writer Loop ═══")
    writer_draft, content_result = await run_writer_loop(
        analyst_output=analyst_output,
        events=events,
        blueprint=blueprint,
        sacred_elements=sacred_elements,
        editorial_angle=editorial_angle,
        max_attempts=max_writer_attempts,
        client=client,
    )
    logger.info(f"   ═══ [{region}] Writer Loop Complete: {content_result.score}/100 ═══")
    
    # Phase 2: Stylist Loop
    logger.info(f"   ═══ [{region}] Starting Stylist Loop ═══")
    styled_draft, style_result = await run_stylist_loop(
        writer_draft=writer_draft,
        blueprint=blueprint,
        sacred_elements=sacred_elements,
        max_attempts=max_stylist_attempts,
        client=client,
    )
    logger.info(f"   ═══ [{region}] Stylist Loop Complete: {style_result.score}/100 ═══")
    
    return writer_draft, styled_draft, content_result, style_result
