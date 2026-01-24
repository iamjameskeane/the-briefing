"""
The Briefing - Phase 1: Data Aggregation

Fetches events from R2 and filters to the past 7 days.
Groups by region and calculates statistics for the clustering phase.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import boto3

# Region type (matches the main worker)
Region = Literal[
    "EUROPE",
    "MIDDLE_EAST", 
    "EAST_ASIA",
    "SOUTHEAST_ASIA",
    "SOUTH_ASIA",
    "CENTRAL_ASIA",
    "OCEANIA",
    "AFRICA",
    "AMERICAS",
    "OTHER",
]

REGIONS: list[Region] = [
    "EUROPE",
    "MIDDLE_EAST",
    "EAST_ASIA",
    "SOUTHEAST_ASIA",
    "SOUTH_ASIA",
    "CENTRAL_ASIA",
    "OCEANIA",
    "AFRICA",
    "AMERICAS",
    "OTHER",
]


@dataclass
class WeekStats:
    """Statistics for the aggregated week."""
    total_events: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[int, int] = field(default_factory=dict)
    by_region: dict[str, int] = field(default_factory=dict)
    avg_sources_per_event: float = 0.0
    high_severity_events: int = 0  # Severity >= 7
    multi_source_events: int = 0   # Events with 2+ sources


@dataclass
class AggregatedWeek:
    """Output of Phase 1: Aggregated week data for clustering."""
    week_start: datetime
    week_end: datetime
    events: list[dict]
    by_region: dict[str, list[dict]]
    stats: WeekStats
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "week_start": self.week_start.isoformat(),
            "week_end": self.week_end.isoformat(),
            "event_count": len(self.events),
            "stats": {
                "total_events": self.stats.total_events,
                "by_category": self.stats.by_category,
                "by_severity": self.stats.by_severity,
                "by_region": self.stats.by_region,
                "avg_sources_per_event": round(self.stats.avg_sources_per_event, 2),
                "high_severity_events": self.stats.high_severity_events,
                "multi_source_events": self.stats.multi_source_events,
            },
            "regions_with_events": list(self.by_region.keys()),
        }


def get_r2_client():
    """Create R2 (S3-compatible) client from environment variables."""
    endpoint_url = os.getenv("R2_ENDPOINT_URL")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    
    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError("R2 environment variables not fully configured")
    
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def fetch_events_from_r2() -> list[dict]:
    """
    Fetch events.json from R2.
    
    Returns:
        List of event dictionaries.
    """
    bucket_name = os.getenv("R2_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("R2_BUCKET_NAME not set")
    
    s3 = get_r2_client()
    
    try:
        response = s3.get_object(Bucket=bucket_name, Key="events.json")
        events = json.loads(response["Body"].read().decode("utf-8"))
        print(f"📦 Fetched {len(events)} events from R2")
        return events
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__):
            print("📭 No events.json found in R2")
            return []
        raise


def parse_event_timestamp(event: dict) -> datetime | None:
    """
    Parse event timestamp to datetime.
    Tries both 'timestamp' and 'last_updated' fields.
    """
    for field in ["last_updated", "timestamp"]:
        ts = event.get(field)
        if not ts:
            continue
        try:
            # Handle Z suffix
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue
    return None


def filter_events_to_week(
    events: list[dict],
    week_end: datetime | None = None,
) -> tuple[list[dict], datetime, datetime]:
    """
    Filter events to the past 7 days.
    
    Args:
        events: List of event dictionaries.
        week_end: End of the week (defaults to now). 
                  For the weekly run, this should be Saturday 23:59 UTC.
    
    Returns:
        Tuple of (filtered_events, week_start, week_end)
    """
    if week_end is None:
        week_end = datetime.now(timezone.utc)
    
    # Ensure timezone aware
    if week_end.tzinfo is None:
        week_end = week_end.replace(tzinfo=timezone.utc)
    
    # Week starts 7 days before end
    week_start = week_end - timedelta(days=7)
    
    filtered = []
    skipped_old = 0
    skipped_future = 0
    skipped_no_ts = 0
    
    for event in events:
        event_dt = parse_event_timestamp(event)
        
        if event_dt is None:
            skipped_no_ts += 1
            continue
            
        if event_dt < week_start:
            skipped_old += 1
            continue
            
        if event_dt > week_end:
            skipped_future += 1
            continue
            
        filtered.append(event)
    
    print(f"📅 Week: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
    print(f"   ✓ {len(filtered)} events in window")
    if skipped_old:
        print(f"   ↳ {skipped_old} older than 7 days")
    if skipped_future:
        print(f"   ↳ {skipped_future} in the future")
    if skipped_no_ts:
        print(f"   ↳ {skipped_no_ts} missing timestamp")
    
    return filtered, week_start, week_end


def group_events_by_region(events: list[dict]) -> dict[str, list[dict]]:
    """
    Group events by geographic region.
    
    Uses the 'region' field that's already set by the main worker.
    """
    by_region: dict[str, list[dict]] = defaultdict(list)
    
    for event in events:
        region = event.get("region", "OTHER")
        by_region[region].append(event)
    
    # Sort each region by severity (descending) then timestamp (descending)
    for region in by_region:
        by_region[region].sort(
            key=lambda e: (
                -e.get("severity", 0),
                e.get("last_updated", e.get("timestamp", ""))
            ),
            reverse=True
        )
    
    print(f"\n📊 Events by region:")
    for region in REGIONS:
        count = len(by_region.get(region, []))
        if count > 0:
            # Get trend indicator based on severity distribution
            high_sev = sum(1 for e in by_region[region] if e.get("severity", 0) >= 7)
            trend = "🔴" if high_sev >= 3 else "🟡" if high_sev >= 1 else "🟢"
            print(f"   {trend} {region}: {count} events ({high_sev} high-severity)")
    
    return dict(by_region)


def calculate_stats(events: list[dict]) -> WeekStats:
    """
    Calculate statistics for the week's events.
    """
    stats = WeekStats()
    stats.total_events = len(events)
    
    stats.by_category = defaultdict(int)
    stats.by_severity = defaultdict(int)
    stats.by_region = defaultdict(int)
    
    total_sources = 0
    
    for event in events:
        # Category breakdown
        category = event.get("category", "UNKNOWN")
        stats.by_category[category] += 1
        
        # Severity breakdown
        severity = event.get("severity", 0)
        stats.by_severity[severity] += 1
        
        if severity >= 7:
            stats.high_severity_events += 1
        
        # Region breakdown
        region = event.get("region", "OTHER")
        stats.by_region[region] += 1
        
        # Source count
        sources = event.get("sources", [])
        source_count = len(sources) if sources else 1
        total_sources += source_count
        
        if source_count >= 2:
            stats.multi_source_events += 1
    
    # Average sources per event
    if events:
        stats.avg_sources_per_event = total_sources / len(events)
    
    # Convert defaultdicts to regular dicts
    stats.by_category = dict(stats.by_category)
    stats.by_severity = dict(stats.by_severity)
    stats.by_region = dict(stats.by_region)
    
    return stats


def aggregate_week(week_end: datetime | None = None) -> AggregatedWeek:
    """
    Main Phase 1 function: Fetch, filter, group, and calculate stats.
    
    Args:
        week_end: End of the week (defaults to now for Saturday runs).
    
    Returns:
        AggregatedWeek with all data needed for Phase 2 clustering.
    """
    print("\n" + "=" * 60)
    print("📊 PHASE 1: DATA AGGREGATION")
    print("=" * 60)
    
    # Fetch from R2
    all_events = fetch_events_from_r2()
    
    if not all_events:
        raise ValueError("No events found in R2. Cannot generate briefing.")
    
    # Filter to week
    events, week_start, week_end_actual = filter_events_to_week(all_events, week_end)
    
    if not events:
        raise ValueError(f"No events found in the past 7 days. Cannot generate briefing.")
    
    # Group by region
    by_region = group_events_by_region(events)
    
    # Calculate statistics
    stats = calculate_stats(events)
    
    # Print summary
    print(f"\n📈 Week Summary:")
    print(f"   Total events: {stats.total_events}")
    print(f"   High-severity (7+): {stats.high_severity_events}")
    print(f"   Multi-source: {stats.multi_source_events}")
    print(f"   Avg sources/event: {stats.avg_sources_per_event:.1f}")
    
    print(f"\n   By category:")
    for cat, count in sorted(stats.by_category.items(), key=lambda x: -x[1]):
        print(f"      {cat}: {count}")
    
    return AggregatedWeek(
        week_start=week_start,
        week_end=week_end_actual,
        events=events,
        by_region=by_region,
        stats=stats,
    )


if __name__ == "__main__":
    """Test aggregation standalone."""
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        result = aggregate_week()
        print("\n" + "=" * 60)
        print("✅ Aggregation complete!")
        print(json.dumps(result.to_dict(), indent=2))
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
