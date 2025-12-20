from ingestion.youtube.fetch import YoutubeFetcher
from ingestion.youtube.normalize import Normalizer
from ingestion.youtube.chunk import Chunker

from core.llm.base import Embedder
from core.vector_store.base import VectorStore


def ingest_youtube_video(
    video_id: str,
    embedder: Embedder,
    vector_store: VectorStore,
    max_chars: int = 2000,
    overlap_ratio: float = 0.1,
) -> None:
    """
    Ingest a YouTube video into the vector store.

    Steps:
    1. Fetch transcript
    2. Normalize segments
    3. Chunk segments
    4. Embed chunks
    5. Store vectors + metadata
    """
    # 1. Fetch raw transcript
    raw_segments = YoutubeFetcher.fetch_transcript(video_id)

    # 2. Normalize
    normalized_segments = Normalizer.normalize_segments(raw_segments)

    if not normalized_segments:
        raise ValueError(f"No usable transcript after normalization: {video_id}")

    # 3. Chunk
    chunks = Chunker.chunk_segments(
        normalized_segments,
        max_chars=max_chars,
        overlap_ratio=overlap_ratio,
    )

    if not chunks:
        raise ValueError(f"No chunks produced for video: {video_id}")

    # 4. Prepare texts + metadata
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

    # 5. Embed and store
    embeddings = embedder.embed(texts)
    vector_store.add(embeddings, metadatas)
