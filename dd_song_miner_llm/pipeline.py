from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .asr import Transcriber
from .ffmpeg import cut_audio, cut_video, extract_audio, get_duration
from .llm import identify_songs
from .merger import build_song_results
from .models import SongResult, TranscriptSegment
from .report import write_reports


def _safe_filename(value: str, fallback: str = "untitled") -> str:
    import re
    import unicodedata
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    return normalized[:120] or fallback


def _format_compact_timecode(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}-{m:02d}-{s:02d}"


def run_pipeline(
    input_video: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[SongResult]:
    input_path = Path(input_video)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audio_dir = out / "01_audio"
    asr_dir = out / "02_asr"
    clips_dir = out / "03_clips"
    reports_dir = out / "04_reports"
    for d in [audio_dir, asr_dir, clips_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[1/4] Extracting audio...")
    source_wav = audio_dir / "source.wav"
    if not source_wav.exists():
        extract_audio(
            input_path, source_wav,
            sample_rate=int(config["audio"]["sample_rate"]),
            channels=int(config["audio"]["channels"]),
        )

    total_duration = get_duration(input_path)

    print("[2/4] Running Whisper ASR...")
    transcript_path = asr_dir / "transcript.json"
    if transcript_path.exists():
        segments = [
            TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in json.loads(transcript_path.read_text(encoding="utf-8"))
        ]
    else:
        transcriber = Transcriber(config)
        segments = transcriber.transcribe(source_wav)
        transcript_path.write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"  Transcribed {len(segments)} segments")

    print("[3/4] Identifying songs with LLM...")
    matches = identify_songs(segments, config)
    print(f"  Found {len(matches)} song matches")

    print("[4/4] Building results and exporting...")
    results = build_song_results(segments, matches, total_duration, config)

    audio_ext = str(config["output"].get("audio_extension", "m4a")).lstrip(".")
    audio_dir_out = clips_dir / "audio"
    video_dir_out = clips_dir / "video"

    for result in results:
        name_bits = [f"{result.index:03d}", result.title]
        if result.artist:
            name_bits.append(result.artist)
        stem = _safe_filename(" - ".join(name_bits))

        if config["output"].get("audio_segments", True):
            try:
                target = audio_dir_out / f"{stem}.{audio_ext}"
                cut_audio(input_path, target, result.start, result.end, copy_codec=True)
                result.audio_path = target
            except Exception as exc:
                result.errors.append(f"audio export failed: {exc}")

        if config["output"].get("video_clips", True):
            try:
                target = video_dir_out / f"{stem}.mp4"
                cut_video(input_path, target, result.start, result.end)
                result.video_path = target
            except Exception as exc:
                result.errors.append(f"video export failed: {exc}")

    write_reports(results, reports_dir)

    manifest = {
        "input_video": str(input_path),
        "total_duration": total_duration,
        "segment_count": len(segments),
        "song_count": len(results),
        "config": {
            "asr_model": config["asr"]["model"],
            "llm_model": config["llm"]["model"],
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return results
