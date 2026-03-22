import logging

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

logger = logging.getLogger(__name__)


class YoutubeTranscriptError(Exception):
    """Base exception for YouTube transcript errors."""

class TranscriptNotAvailableError(YoutubeTranscriptError):
    """Raised when a transcript cannot be fetched for a video."""

class YoutubeFetcher:
    @staticmethod
    def fetch_transcript(video_id: str) -> list[dict]:
        """
        Fetch raw YouTube transcript segments.

        Args:
            video_id: Valid YouTube video ID (11 characters)

        Returns:
            List of dicts with keys: text, start, duration

        Raises:
            TranscriptNotAvailableError: if transcript is unavailable
        """
        try:
            logger.info("YoutubeFetcher.fetch_transcript: video_id=%s", video_id)
            youtube_transcript_api = YouTubeTranscriptApi()
            transcript = youtube_transcript_api.fetch(video_id, languages=["en"])
            snippets = [
                {
                    "text": snippet.text,
                    "start": snippet.start,
                    "duration": snippet.duration,
                }
                for snippet in transcript
            ]
            logger.info(
                "YoutubeFetcher.fetch_transcript: ok video_id=%s snippet_count=%d",
                video_id,
                len(snippets),
            )
            return snippets

        except (TranscriptsDisabled, NoTranscriptFound):
            raise TranscriptNotAvailableError(
                f"No transcript available for video: {video_id}"
            )

        except VideoUnavailable:
            raise TranscriptNotAvailableError(
                f"Video is unavailable: {video_id}"
            )

        except Exception as e:
            raise YoutubeTranscriptError(
                f"Unexpected error fetching transcript for {video_id}: {e}"
            )