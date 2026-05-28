from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .models import SongResult


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


def _alternate_report_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")
