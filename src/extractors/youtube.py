import hashlib
import re
from youtube_transcript_api import YouTubeTranscriptApi
from src.models import Document, SourceType

def _extract_video_id(url: str) -> str:
    patterns = [
        r"youtu\.be/([^?&]+)",
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtube\.com/shorts/([^?&]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")

def fetch_youtube(url: str) -> Document:
    video_id = _extract_video_id(url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    # New API — instantiate the client first
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id)
    full_text = " ".join(seg.text for seg in fetched)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    content_hash = hashlib.sha256(full_text.encode()).hexdigest()
    doc_id = hashlib.sha256(canonical_url.encode()).hexdigest()[:16]

    return Document(
        id=doc_id,
        source_url=canonical_url,
        source_type=SourceType.YOUTUBE,
        title=f"YouTube: {video_id}",
        raw_text=full_text,
        metadata={"video_id": video_id},
        content_hash=content_hash,
    )
