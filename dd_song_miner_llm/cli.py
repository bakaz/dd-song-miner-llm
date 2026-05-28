from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .pipeline import run_pipeline


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

    init_parser = subparsers.add_parser("init-config", help="Generate default config file.")
    init_parser.add_argument("--out", default="config.yaml", help="Output path.")

    return parser


def _generate_config_yaml() -> str:
    lines = [
        "# dd-song-miner-llm 配置文件",
        "",
        "# ASR 配置",
        "asr:",
        "  model: small",
        "  device: auto",
        "  language: null",
        "  beam_size: 5",
        "  vad_filter: true",
        "",
        "# LLM 配置",
        "llm:",
        "  api_key: sk-xxx  # 必填",
        "  api_key_env: null # 可选：从环境变量读取 API key，例如 DEEPSEEK_API_KEY",
        "  base_url: null   # 兼容 OpenAI 格式的自定义 URL",
        "  model: gpt-4o",
        "  temperature: 0.3",
        "  max_tokens: 4096",
        "  max_completion_tokens: null # MiMo 等模型可设为 8192 或更高，优先于 max_tokens",
        "  retry_empty_with_reasoning: true",
        "  reasoning_followup_rounds: 2",
        "  reasoning_followup_max_tokens: 8192",
        "  batch_size: null # null=整段提交；正整数=按 ASR 段数分批",
        "",
        "# 时间 padding",
        "padding:",
        "  before_seconds: 3.0",
        "  after_seconds: 5.0",
        "  min_song_seconds: 15.0",
        "  merge_gap_seconds: 30.0",
        "",
        "# 输出",
        "output:",
        "  video_clips: true",
        "  audio_segments: true",
        "  audio_extension: m4a",
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

    if args.command == "run":
        config = load_config(args.config)

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

        # 检查API key（支持环境变量）
        api_key = config["llm"].get("api_key")
        api_key_env = config["llm"].get("api_key_env")
        if not api_key and api_key_env:
            import os
            api_key = os.environ.get(str(api_key_env), "")
        if not api_key:
            print("Error: LLM API key required. Set in config or --llm-api-key")
            return 1

        output_dir = Path(args.out) if args.out else (
            Path(args.out_root) / f"{Path(args.video).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        results = run_pipeline(Path(args.video), output_dir, config)
        print(f"\nDone! Found {len(results)} songs in: {output_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
