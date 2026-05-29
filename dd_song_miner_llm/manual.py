from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .ffmpeg import cut_audio, cut_video
from .models import SongResult
from .paths import stage_input_for_ffmpeg
from .report import write_reports


def manual_cut(
    run_dir: str | Path,
    config: dict[str, Any],
    csv_path: str | Path | None = None,
    input_video: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[SongResult]:
    run = Path(run_dir)
    source_csv = Path(csv_path) if csv_path else run / "04_reports" / "songs.csv"
    source_video = Path(input_video) if input_video else _input_video_from_manifest(run)
    out = Path(output_dir) if output_dir else run / "05_manual"
    audio_out = out / "audio"
    video_out = out / "video"
    reports_out = out / "reports"
    for path in [audio_out, video_out, reports_out]:
        path.mkdir(parents=True, exist_ok=True)
    source_video = stage_input_for_ffmpeg(source_video, out / "00_input").resolve()

    audio_ext = str(config["output"].get("audio_extension", "m4a")).lstrip(".")
    audio_bitrate_kbps = int(config["output"].get("audio_bitrate_kbps") or 320)
    video_ext = str(config["output"].get("video_extension", "mp4")).lstrip(".")
    video_codec = str(config["output"].get("video_codec", "auto"))

    results = _read_manual_rows(source_csv)
    for result in results:
        stem = _safe_manual_stem(result)
        if config["output"].get("audio_segments", True):
            target = audio_out / f"{stem}.{audio_ext}"
            copy_audio = audio_ext.lower() in {"aac", "m4a"}
            cut_audio(
                source_video,
                target,
                result.start,
                result.end,
                copy_codec=copy_audio,
                bitrate_kbps=audio_bitrate_kbps,
            )
            result.audio_path = target

        if config["output"].get("video_clips", True):
            target = video_out / f"{stem}.{video_ext}"
            cut_video(source_video, target, result.start, result.end, video_codec=video_codec)
            result.video_path = target

    write_reports(results, reports_out)
    return results


def _read_manual_rows(csv_path: Path) -> list[SongResult]:
    results: list[SongResult] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for fallback_index, row in enumerate(reader, start=1):
            start = _parse_time(row.get("start", "0"))
            end = _parse_time(row.get("end", "0"))
            if end <= start:
                raise ValueError(f"Invalid manual time range in row {fallback_index}: {start} -> {end}")
            index = int(row.get("index") or fallback_index)
            results.append(SongResult(
                index=index,
                title=row.get("title") or f"manual_{index:03d}",
                artist=row.get("artist") or "",
                start=start,
                end=end,
                duration=end - start,
                lyrics_snippet=row.get("lyrics_snippet") or "",
                confidence=float(row.get("confidence") or 1.0),
                audio_path=None,
                video_path=None,
                transcript=row.get("transcript") or "",
                errors=[],
            ))
    return results


def _input_video_from_manifest(run_dir: Path) -> Path:
    manifest_path = run_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_video = Path(data["input_video"])
    if input_video.is_absolute():
        return input_video
    cwd_candidate = Path.cwd() / input_video
    if cwd_candidate.exists():
        return cwd_candidate
    run_candidate = run_dir / input_video
    if run_candidate.exists():
        return run_candidate
    return input_video


def _parse_time(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(f"Unsupported time value: {value}")


def _safe_manual_stem(result: SongResult) -> str:
    from .paths import safe_path_part

    bits = [f"{result.index:03d}", result.title]
    if result.artist:
        bits.append(result.artist)
    return safe_path_part(" - ".join(bits), fallback=f"manual_{result.index:03d}")
