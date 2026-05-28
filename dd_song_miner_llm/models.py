from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SongMatch:
    title: str
    artist: str
    lyrics_snippet: str
    segment_indices: list[int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SongResult:
    index: int
    title: str
    artist: str
    start: float
    end: float
    duration: float
    lyrics_snippet: str
    confidence: float
    audio_path: Path | None
    video_path: Path | None
    transcript: str
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["audio_path"] = str(self.audio_path) if self.audio_path else None
        data["video_path"] = str(self.video_path) if self.video_path else None
        return data
