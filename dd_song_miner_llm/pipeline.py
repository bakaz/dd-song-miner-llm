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
from .paths import safe_path_part, stage_input_for_ffmpeg
from .report import write_match_context_reports, write_reports


def _safe_filename(value: str, fallback: str = "untitled") -> str:
    return safe_path_part(value, fallback=fallback)


def _format_compact_timecode(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}-{m:02d}-{s:02d}"


def run_pipeline(
    input_video: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[SongResult]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = stage_input_for_ffmpeg(input_video, out / "00_input").resolve()

    audio_dir = out / "01_audio"
    asr_dir = out / "02_asr"
    llm_dir = asr_dir / "llm"
    clips_dir = out / "03_clips"
    reports_dir = out / "04_reports"
    for d in [audio_dir, asr_dir, llm_dir, clips_dir, reports_dir]:
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
    matches = identify_songs(segments, config, debug_dir=llm_dir)
    (llm_dir / "matches.json").write_text(
        json.dumps([match.to_dict() for match in matches], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_match_context_reports(
        matches,
        segments,
        llm_dir,
        context_segments=int(config["output"].get("match_context_segments", 10)),
    )
    print(f"  Found {len(matches)} song matches")

    print("[4/4] Building results and exporting...")
    results = build_song_results(segments, matches, total_duration, config)

    audio_ext = str(config["output"].get("audio_extension", "m4a")).lstrip(".")
    audio_bitrate_kbps = int(config["output"].get("audio_bitrate_kbps") or 320)
    video_ext = str(config["output"].get("video_extension", "mp4")).lstrip(".")
    video_codec = str(config["output"].get("video_codec", "auto"))
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
                copy_audio = audio_ext.lower() in {"aac", "m4a"}
                cut_audio(
                    input_path,
                    target,
                    result.start,
                    result.end,
                    copy_codec=copy_audio,
                    bitrate_kbps=audio_bitrate_kbps,
                )
                result.audio_path = target
            except Exception as exc:
                result.errors.append(f"audio export failed: {exc}")

        if config["output"].get("video_clips", True):
            try:
                target = video_dir_out / f"{stem}.{video_ext}"
                cut_video(input_path, target, result.start, result.end, video_codec=video_codec)
                result.video_path = target
            except Exception as exc:
                result.errors.append(f"video export failed: {exc}")

    write_reports(results, reports_dir)

    # 输出识别结果摘要
    print(f"\n{'='*60}")
    print(f"识别到 {len(results)} 首歌曲:")
    print(f"{'='*60}")
    for r in results:
        tc_start = f"{int(r.start//3600):02d}:{int((r.start%3600)//60):02d}:{int(r.start%60):02d}"
        tc_end = f"{int(r.end//3600):02d}:{int((r.end%3600)//60):02d}:{int(r.end%60):02d}"
        print(f"\n[{r.index}] {r.title}")
        if r.artist:
            print(f"    歌手: {r.artist}")
        print(f"    时间: {tc_start} - {tc_end} ({r.duration:.1f}s)")
        print(f"    置信度: {r.confidence:.2f}")
        if r.transcript:
            # 显示完整的whisper识别歌词
            print(f"    歌词:")
            for line in r.transcript.split(" "):
                if line.strip():
                    print(f"      {line.strip()}")
    print(f"\n{'='*60}")

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
