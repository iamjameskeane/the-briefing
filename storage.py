"""
The Briefing - Storage Phase

Uploads generated briefing to R2.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import boto3
import requests

# Environment variables
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "")


@dataclass
class StorageResult:
    """Result of storage operation."""
    success: bool
    markdown_url: str
    json_url: str
    error: str | None = None


def get_r2_client():
    """Create R2 (S3-compatible) client."""
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        raise ValueError("R2 environment variables not fully configured")
    
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


def upload_to_r2(
    content: str,
    key: str,
    content_type: str = "text/markdown",
) -> str:
    """
    Upload content to R2.
    
    Args:
        content: String content to upload.
        key: Object key (path).
        content_type: MIME type.
    
    Returns:
        URL of uploaded object.
    """
    s3 = get_r2_client()
    
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType=content_type,
    )
    
    # Construct public URL
    if "r2.cloudflarestorage.com" in R2_ENDPOINT_URL:
        public_url = f"https://{R2_BUCKET_NAME}.r2.dev/{key}"
    else:
        public_url = f"{R2_ENDPOINT_URL}/{R2_BUCKET_NAME}/{key}"
    
    return public_url


def upload_image_to_r2(
    image_data: bytes,
    key: str,
    content_type: str = "image/png",
) -> str:
    """
    Upload image binary data to R2.
    
    Args:
        image_data: Binary image data.
        key: Object key (path).
        content_type: MIME type (default: image/png).
    
    Returns:
        Public URL of uploaded image.
    """
    s3 = get_r2_client()
    
    s3.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=image_data,
        ContentType=content_type,
    )
    
    # Construct public URL
    if "r2.cloudflarestorage.com" in R2_ENDPOINT_URL:
        public_url = f"https://{R2_BUCKET_NAME}.r2.dev/{key}"
    else:
        public_url = f"{R2_ENDPOINT_URL}/{R2_BUCKET_NAME}/{key}"
    
    return public_url


def save_briefing(
    briefing_content: str,
    week_start: datetime,
    week_end: datetime,
    stats: dict,
    judge_result: dict | None = None,
    event_ids: list[str] | None = None,
) -> StorageResult:
    """
    Save briefing to R2.
    
    Saves:
    - briefing/YYYY-MM-DD.md - The briefing markdown
    - briefing/YYYY-MM-DD.json - Metadata
    
    Args:
        briefing_content: Generated markdown content.
        week_start: Start of the week.
        week_end: End of the week.
        stats: Generation statistics.
        judge_result: Judge verification result.
        event_ids: List of event IDs covered.
    
    Returns:
        StorageResult with URLs.
    """
    print("\n" + "=" * 60)
    print("💾 PHASE 5: STORAGE")
    print("=" * 60)
    
    date_str = week_end.strftime("%Y-%m-%d")
    md_key = f"briefing/{date_str}.md"
    json_key = f"briefing/{date_str}.json"
    
    try:
        # Upload markdown
        print(f"   📄 Uploading {md_key}...")
        md_url = upload_to_r2(briefing_content, md_key, "text/markdown")
        
        # Create and upload metadata
        metadata = {
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "markdown_key": md_key,
            "stats": stats,
            "scores": judge_result if judge_result else {},
            "event_ids": event_ids or [],
        }
        
        print(f"   📋 Uploading {json_key}...")
        json_url = upload_to_r2(
            json.dumps(metadata, indent=2),
            json_key,
            "application/json",
        )
        
        print(f"   ✅ Saved to R2")
        print(f"      MD:   {md_url}")
        print(f"      JSON: {json_url}")
        
        # Extract title from content
        title = "Weekly Intelligence Briefing"
        subject_match = briefing_content.split("**SUBJECT:**")
        if len(subject_match) > 1:
            title_line = subject_match[1].split("\n")[0].strip()
            if title_line:
                title = title_line
        
        # Update the index
        update_briefing_index(
            date_str=date_str,
            title=title,
            word_count=stats.get("word_count", len(briefing_content.split())),
            events_analyzed=stats.get("total_events", 0),
        )
        
        return StorageResult(
            success=True,
            markdown_url=md_url,
            json_url=json_url,
        )
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"   ❌ Storage failed: {error_msg}")
        return StorageResult(
            success=False,
            markdown_url="",
            json_url="",
            error=error_msg,
        )


def update_briefing_index(
    date_str: str,
    title: str,
    word_count: int,
    events_analyzed: int,
) -> None:
    """
    Update the Briefing index file with a new briefing.
    
    The index is used by the frontend to list available briefings.
    """
    index_key = "briefing/index.json"
    
    try:
        # Try to fetch existing index
        s3 = get_s3_client()
        try:
            response = s3.get_object(Bucket=R2_BUCKET_NAME, Key=index_key)
            existing = json.loads(response["Body"].read().decode("utf-8"))
            briefings = existing.get("briefings", [])
        except Exception:
            # Index doesn't exist yet
            briefings = []
        
        # Check if this date already exists
        existing_ids = {b["id"] for b in briefings}
        if date_str not in existing_ids:
            briefings.append({
                "id": date_str,
                "date": date_str,
                "title": title,
                "wordCount": word_count,
                "eventsAnalyzed": events_analyzed,
            })
        else:
            # Update existing entry
            for b in briefings:
                if b["id"] == date_str:
                    b["title"] = title
                    b["wordCount"] = word_count
                    b["eventsAnalyzed"] = events_analyzed
                    break
        
        # Sort by date descending
        briefings.sort(key=lambda x: x["date"], reverse=True)
        
        # Keep only last 52 weeks
        briefings = briefings[:52]
        
        # Upload updated index
        index_content = json.dumps({"briefings": briefings}, indent=2)
        upload_to_r2(index_content, index_key, "application/json")
        print(f"   📑 Updated index with {len(briefings)} briefings")
        
    except Exception as e:
        print(f"   ⚠️ Failed to update index: {e}")


def save_failed_draft(
    briefing_content: str,
    week_end: datetime,
    failure_reason: str,
) -> str | None:
    """
    Save a failed draft for manual review.
    
    Args:
        briefing_content: The failed draft content.
        week_end: End of the week.
        failure_reason: Why it failed.
    
    Returns:
        URL if saved, None if storage failed.
    """
    date_str = week_end.strftime("%Y-%m-%d")
    key = f"briefing/{date_str}-failed.md"
    
    # Add failure header
    content = f"""<!-- FAILED DRAFT - DO NOT PUBLISH -->
<!-- Failure Reason: {failure_reason} -->
<!-- Manual review required -->

{briefing_content}
"""
    
    try:
        url = upload_to_r2(content, key, "text/markdown")
        print(f"   💾 Failed draft saved: {url}")
        return url
    except Exception as e:
        print(f"   ⚠️ Could not save failed draft: {e}")
        return None


def update_redis_latest(week_end: datetime) -> bool:
    """
    Update Redis with latest Briefing date.
    
    Args:
        week_end: End of the week.
    
    Returns:
        True if updated.
    """
    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "")
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    
    if not redis_url or not redis_token:
        print("   ⚠️ Redis not configured, skipping latest update")
        return False
    
    date_str = week_end.strftime("%Y-%m-%d")
    
    try:
        response = requests.post(
            f"{redis_url}/set/briefing:latest/{date_str}",
            headers={"Authorization": f"Bearer {redis_token}"},
            timeout=5,
        )
        
        if response.ok:
            print(f"   📌 Redis briefing:latest = {date_str}")
            return True
        else:
            print(f"   ⚠️ Redis update failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ⚠️ Redis update failed: {e}")
        return False
