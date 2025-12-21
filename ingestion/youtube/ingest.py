import re
from urllib.parse import urlparse, parse_qs

from ingestion.youtube.fetch import YoutubeFetcher
from ingestion.youtube.normalize import Normalizer
from ingestion.youtube.chunk import Chunker

from core.llm.base import Embedder
from core.vector_store.base import VectorStore


class InvalidVideoIdError(Exception):
    """Raised when the provided video ID or URL is invalid."""


def extract_video_id(input_str: str) -> str:
    """
    Extract and validate YouTube video ID from URL or direct ID.
    
    Args:
        input_str: YouTube video ID or URL
    
    Returns:
        Valid 11-character video ID
        
    Raises:
        InvalidVideoIdError: if the input is not a valid YouTube URL or video ID
    """
    if not input_str:
        raise InvalidVideoIdError("Empty video ID or URL provided")
    
    trimmed = input_str.strip()
    
    # Check if it's already a valid video ID (11 chars, alphanumeric + dashes/underscores)
    if re.match(r"^[\w-]{11}$", trimmed):
        return trimmed
    
    # Check if it's a YouTube URL
    youtube_regex = r"^(?:https?://)?(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/.+$"
    if not re.match(youtube_regex, trimmed, re.IGNORECASE):
        raise InvalidVideoIdError(f"Invalid YouTube URL format: {trimmed}")
    
    try:
        # Add protocol if missing for URL parsing
        if not trimmed.startswith(("http://", "https://")):
            trimmed = f"https://{trimmed}"
        
        parsed = urlparse(trimmed)
        
        # Handle ?v=VIDEO_ID
        if parsed.query:
            query_params = parse_qs(parsed.query)
            if "v" in query_params:
                video_id = query_params["v"][0]
                if re.match(r"^[\w-]{11}$", video_id):
                    return video_id
        
        # Handle /embed/VIDEO_ID, /live/VIDEO_ID, or youtu.be/VIDEO_ID
        path_match = re.search(r"(?:/(?:embed|live)/|youtu\.be/)([\w-]{11})", parsed.path, re.IGNORECASE)
        if path_match:
            video_id = path_match.group(1)
            if re.match(r"^[\w-]{11}$", video_id):
                return video_id
        
        raise InvalidVideoIdError(f"Could not extract valid video ID from: {input_str}")
    except Exception as e:
        if isinstance(e, InvalidVideoIdError):
            raise
        raise InvalidVideoIdError(f"Error parsing YouTube URL: {e}")


def ingest_youtube_video(
    video_id_or_url: str,
    embedder: Embedder,
    vector_store: VectorStore,
    max_chars: int = 2000,
    overlap_ratio: float = 0.1,
) -> None:
    """
    Ingest a YouTube video into the vector store.

    Args:
        video_id_or_url: YouTube video ID or URL (supports various formats)
        embedder: Embedder instance for generating embeddings
        vector_store: VectorStore instance for storing vectors
        max_chars: Maximum characters per chunk
        overlap_ratio: Ratio of segments to overlap between chunks

    Steps:
    1. Clear vector store before ingesting
    2. Extract and validate video ID from URL or direct ID
    3. Fetch transcript
    4. Normalize segments
    5. Chunk segments
    6. Embed chunks
    7. Store vectors + metadata
    """

    # 1. Clear vector store before ingesting
    vector_store.clear()

    # 2. Extract and validate video ID
    video_id = extract_video_id(video_id_or_url)
    
    # 3. Fetch raw transcript
    raw_segments = YoutubeFetcher.fetch_transcript(video_id)

    # 4. Normalize
    normalized_segments = Normalizer.normalize_segments(raw_segments)

    if not normalized_segments:
        raise ValueError(f"No usable transcript after normalization: {video_id}")

    # 5. Chunk
    chunks = Chunker.chunk_segments(
        normalized_segments,
        max_chars=max_chars,
        overlap_ratio=overlap_ratio,
    )

    if not chunks:
        raise ValueError(f"No chunks produced for video: {video_id}")

    # 6. Prepare texts + metadata
    texts = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        texts.append(chunk["text"])
        metadatas.append({
            "video_id": video_id,
            "chunk_index": idx,
            "text": chunk["text"],
            "start": chunk["start"],
            "end": chunk["end"],
            "source": "youtube",
        })

    # 7. Embed and store
    embeddings = embedder.embed(texts)
    vector_store.add(embeddings, metadatas)
