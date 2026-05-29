from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise FFmpegError(
                "ffmpeg not found. Install FFmpeg or: pip install imageio-ffmpeg"
            ) from exc
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise FFmpegError(f"Binary not found: {name}")


def run_command(args: list[str], timeout: int = 3600) -> None:
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FFmpegError(f"Command failed: {' '.join(args)}\n{detail}")


def run_command_with_fallback(commands: list[list[str]], timeout: int = 3600) -> None:
    errors: list[str] = []
    for args in commands:
        try:
            run_command(args, timeout=timeout)
            return
        except FFmpegError as exc:
            errors.append(str(exc))
    raise FFmpegError("\n\n".join(errors))


def extract_audio(
    input_video: str | Path,
    output_wav: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        ffmpeg_bin, "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(output),
    ])
    return output


def cut_audio(
    input_media: str | Path,
    output_audio: str | Path,
    start: float,
    end: float,
    copy_codec: bool = False,
    bitrate_kbps: int | None = None,
) -> Path:
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_audio)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin, "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(input_media)]
    if copy_codec:
        cmd.extend(["-c:a", "copy", "-avoid_negative_ts", "make_zero"])
    else:
        cmd.extend(_audio_encode_args(output, bitrate_kbps=bitrate_kbps))
    cmd.append(str(output))
    run_command(cmd)
    return output


def _audio_encode_args(output_audio: Path, bitrate_kbps: int | None = None) -> list[str]:
    ext = output_audio.suffix.lower().lstrip(".")
    bitrate = max(1, int(bitrate_kbps or 320))
    if ext == "wav":
        return ["-vn", "-acodec", "pcm_s16le"]
    if ext == "mp3":
        return ["-vn", "-acodec", "libmp3lame", "-b:a", f"{bitrate}k"]
    if ext in {"m4a", "aac"}:
        return ["-vn", "-acodec", "aac", "-b:a", f"{bitrate}k"]
    if ext == "flac":
        return ["-vn", "-acodec", "flac"]
    if ext == "opus":
        return ["-vn", "-acodec", "libopus", "-b:a", f"{bitrate}k"]
    return ["-vn"]


def cut_video(
    input_video: str | Path,
    output_video: str | Path,
    start: float,
    end: float,
    video_codec: str = "auto",
) -> Path:
    ffmpeg_bin = require_binary("ffmpeg")
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = [
        ffmpeg_bin, "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(input_video),
        "-map", "0:v:0?",
        "-map", "0:a:0?",
    ]
    commands = [base + args + ["-c:a", "copy", "-avoid_negative_ts", "make_zero", str(output)]
                for args in _video_encode_arg_candidates(ffmpeg_bin, video_codec)]
    run_command_with_fallback(commands)
    return output


def _video_encode_arg_candidates(ffmpeg_bin: str, video_codec: str = "auto") -> list[list[str]]:
    codec = (video_codec or "auto").lower()
    if codec == "copy":
        return [["-c:v", "copy"]]
    if codec in {"cpu", "libx264"}:
        return [["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]]
    if codec == "nv":
        return [["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"]]
    if codec == "intel":
        return [["-c:v", "h264_qsv", "-global_quality", "20"]]
    if codec == "amd":
        return [["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "20", "-qp_p", "20", "-qp_b", "20"]]

    encoders = detect_video_encoders(ffmpeg_bin)
    candidates: list[list[str]] = []
    if "h264_nvenc" in encoders:
        candidates.append(["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"])
    if "h264_qsv" in encoders:
        candidates.append(["-c:v", "h264_qsv", "-global_quality", "20"])
    if "h264_amf" in encoders:
        candidates.append(["-c:v", "h264_amf", "-quality", "quality", "-qp_i", "20", "-qp_p", "20", "-qp_b", "20"])
    candidates.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
    return candidates


@lru_cache(maxsize=8)
def detect_video_encoders(ffmpeg_bin: str | None = None) -> set[str]:
    exe = ffmpeg_bin or require_binary("ffmpeg")
    completed = subprocess.run(
        [exe, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    return set(re.findall(r"\b(h264_nvenc|h264_qsv|h264_amf|libx264)\b", text))


def get_duration(input_media: str | Path) -> float:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        completed = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(input_media)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            return float(completed.stdout.strip())

    ffmpeg_bin = require_binary("ffmpeg")
    completed = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", str(input_media)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    duration = _parse_ffmpeg_duration(completed.stderr)
    if duration is None:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FFmpegError(f"Could not read media duration: {input_media}\n{detail}")
    return duration


def _parse_ffmpeg_duration(text: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds
