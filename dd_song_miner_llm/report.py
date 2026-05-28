from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .models import SongMatch, SongResult, TranscriptSegment


def _format_timecode(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_reports(results: list[SongResult], output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "songs.csv"
    json_path = out / "songs.json"

    try:
        csv_file = csv_path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        csv_path = _alternate_report_path(csv_path)
        csv_file = csv_path.open("w", encoding="utf-8-sig", newline="")

    with csv_file as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "start", "end", "duration_seconds",
            "title", "artist", "confidence",
            "audio_path", "video_path", "transcript", "errors",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "index": r.index,
                "start": _format_timecode(r.start),
                "end": _format_timecode(r.end),
                "duration_seconds": round(r.duration, 3),
                "title": r.title,
                "artist": r.artist,
                "confidence": r.confidence,
                "audio_path": str(r.audio_path) if r.audio_path else "",
                "video_path": str(r.video_path) if r.video_path else "",
                "transcript": r.transcript,
                "errors": " | ".join(r.errors),
            })

    try:
        json_file = json_path.open("w", encoding="utf-8")
    except PermissionError:
        json_path = _alternate_report_path(json_path)
        json_file = json_path.open("w", encoding="utf-8")

    with json_file as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def write_match_context_reports(
    matches: list[SongMatch],
    segments: list[TranscriptSegment],
    output_dir: str | Path,
    context_segments: int = 10,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "match_context.json"
    csv_path = out / "match_context.csv"

    rows = []
    payload = []
    for match_index, match in enumerate(matches, start=1):
        valid = sorted({i for i in match.segment_indices if 0 <= i < len(segments)})
        if not valid:
            continue
        first = valid[0]
        last = valid[-1]
        context_start = max(0, first - context_segments)
        context_end = min(len(segments) - 1, last + context_segments)

        context_items = []
        for idx in range(context_start, context_end + 1):
            segment = segments[idx]
            item = {
                "segment_index": idx,
                "start": segment.start,
                "end": segment.end,
                "start_timecode": _format_timecode(segment.start),
                "end_timecode": _format_timecode(segment.end),
                "is_match": idx in valid,
                "text": segment.text,
            }
            context_items.append(item)
            rows.append({
                "match_index": match_index,
                "title": match.title,
                "artist": match.artist,
                "confidence": match.confidence,
                "match_start": _format_timecode(segments[first].start),
                "match_end": _format_timecode(segments[last].end),
                **item,
            })

        payload.append({
            "match_index": match_index,
            "title": match.title,
            "artist": match.artist,
            "lyrics_snippet": match.lyrics_snippet,
            "confidence": match.confidence,
            "segment_indices": valid,
            "start": segments[first].start,
            "end": segments[last].end,
            "start_timecode": _format_timecode(segments[first].start),
            "end_timecode": _format_timecode(segments[last].end),
            "matched_segments": [context_items[i - context_start] for i in valid],
            "context_segments": context_items,
        })

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "match_index", "title", "artist", "confidence", "match_start", "match_end",
            "segment_index", "start", "end", "start_timecode", "end_timecode", "is_match", "text",
        ])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, json_path


def _alternate_report_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")
