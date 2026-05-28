from __future__ import annotations

import re
from typing import Any

from .models import SongMatch, SongResult, TranscriptSegment


def _is_speech_segment(text: str) -> bool:
    """判断是否是说话/聊天内容（而非歌词）"""
    speech_patterns = [
        r"谢谢", r"感谢", r"再见", r"拜拜", r"下次", r"直播间",
        r"大家好", r"hello", r"hi", r"嗯", r"好的", r"对",
        r"thank", r"bye", r"see you", r"next time",
    ]
    text_lower = text.lower()
    for pattern in speech_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def _find_song_boundary(
    segments: list[TranscriptSegment],
    song_start_idx: int,
    direction: str,
    max_search: int = 5,
) -> float:
    """向前后搜索歌曲的实际边界，跳过说话内容"""
    if direction == "before":
        # 向前搜索，找到第一个非说话内容的位置
        for i in range(song_start_idx - 1, max(song_start_idx - max_search - 1, -1), -1):
            if i < 0:
                return 0.0
            if not _is_speech_segment(segments[i].text):
                return segments[i].start
        return segments[max(0, song_start_idx - max_search)].start
    else:
        # 向后搜索，找到最后一个非说话内容的位置
        for i in range(song_start_idx + 1, min(song_start_idx + max_search + 1, len(segments))):
            if i >= len(segments):
                return segments[-1].end
            if not _is_speech_segment(segments[i].text):
                return segments[i].end
        return segments[min(len(segments) - 1, song_start_idx + max_search)].end


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
        # 使用智能边界检测
        song_start = song["start"]
        song_end = song["end"]
        
        # 检查歌曲前面的内容，避免包含说话
        before_boundary = _find_song_boundary(segments, song["segment_start_idx"], "before")
        start = max(before_boundary, song_start - before_pad)
        
        # 检查歌曲后面的内容，避免包含说话
        after_boundary = _find_song_boundary(segments, song["segment_end_idx"], "after")
        end = min(after_boundary, song_end + after_pad)
        
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
