from __future__ import annotations

from typing import Any

from .models import SongMatch, SongResult, TranscriptSegment


def _padding_bounds(
    segments: list[TranscriptSegment],
    song_start_idx: int,
    song_end_idx: int,
    total_duration: float,
) -> tuple[float, float]:
    before_limit = segments[song_start_idx - 1].end if song_start_idx > 0 else 0.0
    after_limit = segments[song_end_idx + 1].start if song_end_idx + 1 < len(segments) else total_duration
    return before_limit, after_limit


def _merge_adjacent_songs(
    songs: list[dict[str, Any]],
    merge_gap: float,
) -> list[dict[str, Any]]:
    if not songs:
        return []

    sorted_songs = sorted(songs, key=lambda s: s["start"])
    merged: list[dict[str, Any]] = [sorted_songs[0]]

    for song in sorted_songs[1:]:
        prev = merged[-1]
        if song["start"] - prev["end"] <= merge_gap and song["title"] == prev["title"]:
            prev["end"] = max(prev["end"], song["end"])
            prev["segment_end_idx"] = max(prev["segment_end_idx"], song["segment_end_idx"])
            prev["confidence"] = max(prev["confidence"], song["confidence"])
            prev["transcript"] += " " + song["transcript"]
        else:
            merged.append(song)

    return merged


def build_song_results(
    segments: list[TranscriptSegment],
    matches: list[SongMatch],
    total_duration: float,
    config: dict[str, Any],
) -> list[SongResult]:
    padding_config = config["padding"]
    before_pad = float(padding_config.get("before_seconds", 3.0))
    after_pad = float(padding_config.get("after_seconds", 5.0))
    min_duration = float(padding_config.get("min_song_seconds", 15.0))
    merge_gap = float(padding_config.get("merge_gap_seconds", 30.0))

    raw_songs: list[dict[str, Any]] = []

    for match in matches:
        if not match.segment_indices:
            continue

        valid_indices = [i for i in match.segment_indices if 0 <= i < len(segments)]
        if not valid_indices:
            continue

        start = segments[min(valid_indices)].start
        end = segments[max(valid_indices)].end
        transcript = " ".join(segments[i].text for i in valid_indices)

        raw_songs.append({
            "title": match.title,
            "artist": match.artist,
            "start": start,
            "end": end,
            "segment_start_idx": min(valid_indices),
            "segment_end_idx": max(valid_indices),
            "confidence": match.confidence,
            "transcript": transcript,
            "lyrics_snippet": match.lyrics_snippet,
        })

    merged = _merge_adjacent_songs(raw_songs, merge_gap)

    results: list[SongResult] = []
    for i, song in enumerate(merged):
        song_start = song["start"]
        song_end = song["end"]

        before_limit, after_limit = _padding_bounds(
            segments,
            song["segment_start_idx"],
            song["segment_end_idx"],
            total_duration,
        )
        start = min(song_start, max(before_limit, song_start - before_pad))
        end = max(song_end, min(after_limit, song_end + after_pad))

        # 确保不超出总时长
        start = max(0.0, start)
        end = min(total_duration, end)
        
        duration = end - start

        if duration < min_duration:
            continue

        results.append(SongResult(
            index=i + 1,
            title=song["title"],
            artist=song["artist"],
            start=start,
            end=end,
            duration=duration,
            lyrics_snippet=song.get("lyrics_snippet", ""),
            confidence=song["confidence"],
            audio_path=None,
            video_path=None,
            transcript=song["transcript"],
            errors=[],
        ))

    return results
