from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .asr import Transcriber
from .ffmpeg import cut_audio, cut_video, extract_audio, get_duration
from .llm import identify_songs
from .merger import build_song_results
from .models import SongMatch, SongResult, TranscriptSegment
from .paths import safe_path_part, stage_input_for_ffmpeg
from .report import write_match_context_reports, write_reports


def _safe_filename(value: str, fallback: str = "untitled") -> str:
    return safe_path_part(value, fallback=fallback)


def _format_compact_timecode(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}-{m:02d}-{s:02d}"


def _check_previous_run(out: Path, input_path: Path) -> dict[str, Any] | None:
    """检查是否有可复用的上次运行结果。返回 progress 或 None。"""
    progress_path = out / "progress.json"
    if not progress_path.exists():
        return None
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        prev_input = progress.get("input_video", "")
        if Path(prev_input).resolve() == input_path.resolve():
            return progress
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_progress(out: Path, input_path: Path, step: str, data: dict[str, Any] | None = None) -> None:
    """保存当前步骤的进度。"""
    progress_path = out / "progress.json"
    try:
        progress = {}
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["input_video"] = str(input_path)
        progress["last_completed_step"] = step
        if data:
            progress[step] = data
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _load_previous_segments(asr_dir: Path) -> list[TranscriptSegment] | None:
    """加载之前的 ASR 转写结果。"""
    transcript_path = asr_dir / "transcript.json"
    if not transcript_path.exists():
        return None
    try:
        return [
            TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in json.loads(transcript_path.read_text(encoding="utf-8"))
        ]
    except (json.JSONDecodeError, OSError):
        return None


def _load_previous_matches(llm_dir: Path) -> list[SongMatch] | None:
    """加载之前的 LLM 识别结果。"""
    matches_path = llm_dir / "matches.json"
    if not matches_path.exists():
        return None
    try:
        return [
            SongMatch(
                title=m["title"],
                artist=m.get("artist", ""),
                lyrics_snippet=m.get("lyrics_snippet", ""),
                segment_indices=m.get("segment_indices", []),
                confidence=m.get("confidence", 0.5),
            )
            for m in json.loads(matches_path.read_text(encoding="utf-8"))
        ]
    except (json.JSONDecodeError, OSError, KeyError):
        return None


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

    # 检查是否有可复用的上次运行结果
    prev_progress = _check_previous_run(out, input_path)
    reuse_audio = False
    reuse_asr = False
    reuse_llm = False

    if prev_progress:
        last_step = prev_progress.get("last_completed_step", "")
        print(f"[info] 检测到上次运行结果（完成到 {last_step}），检查可复用的部分...")
        reuse_audio = last_step in ("audio", "asr", "llm", "done") and (audio_dir / "source.wav").exists()
        reuse_asr = last_step in ("asr", "llm", "done") and (asr_dir / "transcript.json").exists()
        reuse_llm = last_step in ("llm", "done") and (llm_dir / "matches.json").exists()
        print(f"  音频提取: {'复用' if reuse_audio else '需要重新运行'}")
        print(f"  ASR 转写: {'复用' if reuse_asr else '需要重新运行'}")
        print(f"  LLM 识别: {'复用' if reuse_llm else '需要重新运行'}")

    # Step 1: 音频提取
    source_wav = audio_dir / "source.wav"
    if reuse_audio:
        print("[1/4] 音频提取: 复用已有结果")
    else:
        print("[1/4] Extracting audio...")
        extract_audio(
            input_path, source_wav,
            sample_rate=int(config["audio"]["sample_rate"]),
            channels=int(config["audio"]["channels"]),
        )
    _save_progress(out, input_path, "audio")

    total_duration = get_duration(input_path)

    # Step 2: ASR 转写
    if reuse_asr:
        print("[2/4] ASR 转写: 复用已有结果")
        segments = _load_previous_segments(asr_dir)
        if segments is None:
            print("  [warn] 无法加载之前的 ASR 结果，重新运行...")
            reuse_asr = False

    if not reuse_asr:
        print("[2/4] Running Whisper ASR...")
        transcriber = Transcriber(config)
        segments = transcriber.transcribe(source_wav)
        transcript_path = asr_dir / "transcript.json"
        transcript_path.write_text(
            json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _save_progress(out, input_path, "asr")

    print(f"  Transcribed {len(segments)} segments")

    # Step 3: LLM 识别
    if reuse_llm:
        print("[3/4] LLM 识别: 复用已有结果")
        matches = _load_previous_matches(llm_dir)
        if matches is None:
            print("  [warn] 无法加载之前的 LLM 结果，重新运行...")
            reuse_llm = False

    if not reuse_llm:
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
    _save_progress(out, input_path, "llm")
    print(f"  Found {len(matches)} song matches")

    print("[4/4] Building results and exporting...")
    results = build_song_results(segments, matches, total_duration, config)

    audio_ext = str(config["output"].get("audio_extension", "m4a")).lstrip(".")
    audio_bitrate_kbps = int(config["output"].get("audio_bitrate_kbps") or 320)
    video_ext = str(config["output"].get("video_extension", "mp4")).lstrip(".")
    video_codec = str(config["output"].get("video_codec", "copy"))
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
    _save_progress(out, input_path, "export")

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
    _save_progress(out, input_path, "done")

    return results
