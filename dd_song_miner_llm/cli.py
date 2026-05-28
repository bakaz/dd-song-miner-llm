from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .ffmpeg import detect_ffmpeg_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dd-song-miner-llm",
        description="Extract songs using Whisper ASR + LLM identification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Process a video file.")
    run_parser.add_argument("video", help="Input video file.")
    run_parser.add_argument("--out", default=None, help="Output directory.")
    run_parser.add_argument("--out-root", default="runs", help="Root for auto-created runs.")
    run_parser.add_argument("--config", default=None, help="YAML config file.")
    run_parser.add_argument("--asr-model", default=None, help="Whisper model (tiny/base/small/medium/large-v3).")
    run_parser.add_argument("--asr-language", default=None, help="ASR language hint (zh/ja/en).")
    run_parser.add_argument("--llm-model", default=None, help="LLM model name (comma-separated for multiple).")
    run_parser.add_argument("--llm-api-key", default=None, help="LLM API key (comma-separated for multiple).")
    run_parser.add_argument("--llm-base-url", default=None, help="LLM API base URL (comma-separated for multiple).")
    run_parser.add_argument("--padding-before", type=float, default=None, help="Padding before song (seconds).")
    run_parser.add_argument("--padding-after", type=float, default=None, help="Padding after song (seconds).")
    run_parser.add_argument("--no-video-clips", action="store_true", help="Skip video clip export.")
    run_parser.add_argument("--export-audio", default=None, help="Audio export format (mp3, m4a, wav, etc).")
    run_parser.add_argument("--export-video", default=None, help="Video export format (mp4, mkv, etc).")
    run_parser.add_argument("--video-codec", default=None, help="Video codec: copy, auto, nv, intel, amd, cpu.")
    run_parser.add_argument("--audio-bitrate-kbps", type=int, default=None, help="Audio transcode bitrate.")

    batch_parser = subparsers.add_parser("batch-run", help="Process videos under a folder tree once per folder.")
    batch_parser.add_argument("input_root", help="Folder to scan recursively.")
    batch_parser.add_argument("--result-root", required=True, help="Where completed run outputs are copied.")
    batch_parser.add_argument("--work-root", default="runs/batch", help="Local working root for pipeline outputs.")
    batch_parser.add_argument("--config", default=None, help="YAML config file.")
    batch_parser.add_argument("--marker", default=".dd_song_miner_done.json", help="Done marker file written into each processed source folder.")
    batch_parser.add_argument("--extensions", default=None, help="Comma-separated video extensions, e.g. mp4,mkv,flv.")
    batch_parser.add_argument("--video-codec", default=None, help="Video codec: copy, auto, nv, intel, amd, cpu.")
    batch_parser.add_argument("--audio-bitrate-kbps", type=int, default=None, help="Audio transcode bitrate.")

    manual_parser = subparsers.add_parser("manual-cut", help="Cut clips from an edited songs.csv in an existing run folder.")
    manual_parser.add_argument("run_dir", help="Existing run output directory.")
    manual_parser.add_argument("--csv", default=None, help="Edited songs.csv path. Defaults to RUN_DIR/04_reports/songs.csv.")
    manual_parser.add_argument("--video", default=None, help="Input video override. Defaults to manifest input_video.")
    manual_parser.add_argument("--out", default=None, help="Manual output directory. Defaults to RUN_DIR/05_manual.")
    manual_parser.add_argument("--config", default=None, help="YAML config file.")
    manual_parser.add_argument("--video-codec", default=None, help="Video codec: copy, auto, nv, intel, amd, cpu.")
    manual_parser.add_argument("--audio-bitrate-kbps", type=int, default=None, help="Audio transcode bitrate.")

    init_parser = subparsers.add_parser("init-config", help="Generate default config file.")
    init_parser.add_argument("--out", default="config.yaml", help="Output path.")

    info_parser = subparsers.add_parser("ffmpeg-info", help="Show GPU and FFmpeg encoder detection.")
    info_parser.add_argument("--ffmpeg", default=None, help="Optional ffmpeg executable path.")

    return parser


def _generate_config_yaml() -> str:
    lines = [
        "# dd-song-miner-llm 配置文件",
        "",
        "# 音频预处理",
        "audio:",
        "  sample_rate: 16000",
        "  channels: 1",
        "",
        "# ASR 配置",
        "asr:",
        "  model: small",
        "  device: auto",
        "  compute_type: default",
        "  language: null",
        "  beam_size: 5",
        "  vad_filter: true",
        "  initial_prompt: null",
        "",
        "# LLM 配置",
        "llm:",
        "  api_key: null # 不建议写明文 key；优先使用 api_key_env",
        "  api_key_env: LLM_API_KEY # 可选：从环境变量读取 API key，例如 DEEPSEEK_API_KEY",
        "  base_url: null # 兼容 OpenAI 格式的自定义 URL",
        "  model: gpt-4o",
        "  temperature: 0.1",
        "  max_tokens: 8192",
        "  max_completion_tokens: null # MiMo 等模型可设为 8192 或更高，优先于 max_tokens",
        "  retry_empty_with_reasoning: true",
        "  reasoning_followup_rounds: 5",
        "  reasoning_followup_max_tokens: 32768",
        "  batch_size: null # null=整段提交；正整数=按 ASR 段数分批",
        "  use_tools: true",
        "  verify_with_search: true",
        "  fallbacks: []",
        "",
        "# 时间 padding",
        "padding:",
        "  before_seconds: 3.0",
        "  after_seconds: 15.0",
        "  after_next_asr_end_guard_seconds: 2.0",
        "  min_song_seconds: 30.0",
        "  merge_gap_seconds: 35.0",
        "",
        "# 输出",
        "output:",
        "  video_clips: true",
        "  audio_segments: true",
        "  audio_extension: m4a",
        "  audio_bitrate_kbps: 320",
        "  video_extension: mp4",
        "  video_codec: copy # copy=复制原视频流；auto=nv > intel > amd > cpu",
        "  match_context_segments: 10",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-config":
        content = _generate_config_yaml()
        out_path = Path(args.out)
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote config: {out_path}")
        return 0

    if args.command == "ffmpeg-info":
        _print_ffmpeg_info(args.ffmpeg)
        return 0

    if args.command == "run":
        from .pipeline import run_pipeline

        config = load_config(args.config)

        _apply_run_overrides(config, args)

        # 检查API key（支持环境变量）
        if not _has_api_key(config):
            print("Error: LLM API key required. Set in config or --llm-api-key")
            return 1

        output_dir = Path(args.out) if args.out else (
            Path(args.out_root) / Path(args.video).stem
        )

        results = run_pipeline(Path(args.video), output_dir, config)
        print(f"\nDone! Found {len(results)} songs in: {output_dir}")
        return 0

    if args.command == "batch-run":
        from .batch import run_batch

        config = load_config(args.config)
        _apply_output_overrides(config, args)
        if not _has_api_key(config):
            print("Error: LLM API key required. Set in config or environment")
            return 1
        extensions = None
        if args.extensions:
            extensions = {item.strip() for item in args.extensions.split(",") if item.strip()}
        runs = run_batch(
            args.input_root,
            args.result_root,
            args.work_root,
            config,
            marker_name=args.marker,
            extensions=extensions,
        )
        print(f"\nDone! Batch produced {len(runs)} run records.")
        return 0

    if args.command == "manual-cut":
        from .manual import manual_cut

        config = load_config(args.config)
        _apply_output_overrides(config, args)
        results = manual_cut(
            args.run_dir,
            config,
            csv_path=args.csv,
            input_video=args.video,
            output_dir=args.out,
        )
        print(f"\nDone! Manual cut produced {len(results)} songs.")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _apply_run_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.asr_model:
        config["asr"]["model"] = args.asr_model
    if args.asr_language:
        config["asr"]["language"] = args.asr_language
    if args.llm_model:
        config["llm"]["model"] = args.llm_model
    if args.llm_api_key:
        config["llm"]["api_key"] = args.llm_api_key
    if args.llm_base_url:
        config["llm"]["base_url"] = args.llm_base_url
    if args.padding_before is not None:
        config["padding"]["before_seconds"] = args.padding_before
    if args.padding_after is not None:
        config["padding"]["after_seconds"] = args.padding_after
    if args.no_video_clips:
        config["output"]["video_clips"] = False
    if args.export_audio:
        config["output"]["audio_segments"] = True
        config["output"]["audio_extension"] = args.export_audio.lstrip(".")
    if args.export_video:
        config["output"]["video_clips"] = True
        config["output"]["video_extension"] = args.export_video.lstrip(".")
    _apply_output_overrides(config, args)


def _apply_output_overrides(config: dict, args: argparse.Namespace) -> None:
    if getattr(args, "video_codec", None):
        config["output"]["video_codec"] = args.video_codec
    if getattr(args, "audio_bitrate_kbps", None) is not None:
        config["output"]["audio_bitrate_kbps"] = args.audio_bitrate_kbps


def _has_api_key(config: dict) -> bool:
    api_key = config["llm"].get("api_key")
    api_key_env = config["llm"].get("api_key_env")
    if not api_key and api_key_env:
        import os
        api_key = os.environ.get(str(api_key_env), "")
    return bool(api_key)


def _print_ffmpeg_info(ffmpeg_bin: str | None = None) -> None:
    info = detect_ffmpeg_environment(ffmpeg_bin)
    print(f"FFmpeg: {info['ffmpeg']}")

    gpus = list(info["gpus"])
    if gpus:
        print("GPU:")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("GPU: not detected")

    hwaccels = list(info["hwaccels"])
    print("FFmpeg hwaccels: " + (", ".join(hwaccels) if hwaccels else "none detected"))

    encoders = list(info["video_encoders"])
    print("FFmpeg H.264 encoders: " + (", ".join(encoders) if encoders else "none detected"))

    auto_order = list(info["auto_reencode_order"])
    print("Auto re-encode order: " + (" > ".join(auto_order) if auto_order else "none"))
    print("Recommended for fastest lossless-quality clipping: --video-codec copy")


if __name__ == "__main__":
    raise SystemExit(main())
