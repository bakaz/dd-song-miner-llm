from __future__ import annotations

import unittest

from pathlib import Path

from dd_song_miner_llm.ffmpeg import _audio_encode_args, _parse_ffmpeg_duration, _video_encode_arg_candidates


class FFmpegDurationTests(unittest.TestCase):
    def test_parse_duration_from_ffmpeg_stderr(self) -> None:
        text = "Input #0, mp4, from 'video.mp4':\n  Duration: 01:02:03.45, start: 0.000000"

        self.assertAlmostEqual(_parse_ffmpeg_duration(text), 3723.45)

    def test_parse_duration_returns_none_when_missing(self) -> None:
        self.assertIsNone(_parse_ffmpeg_duration("no duration here"))

    def test_mp3_audio_export_transcodes_instead_of_copying(self) -> None:
        args = _audio_encode_args(Path("clip.mp3"))

        self.assertIn("libmp3lame", args)
        self.assertIn("320k", args)
        self.assertNotIn("copy", args)

    def test_transcode_bitrate_can_be_configured(self) -> None:
        args = _audio_encode_args(Path("clip.m4a"), bitrate_kbps=384)

        self.assertIn("aac", args)
        self.assertIn("384k", args)

    def test_explicit_gpu_video_encoder_order_options(self) -> None:
        self.assertIn("h264_nvenc", _video_encode_arg_candidates("ffmpeg", "nv")[0])
        self.assertIn("h264_qsv", _video_encode_arg_candidates("ffmpeg", "intel")[0])
        self.assertIn("h264_amf", _video_encode_arg_candidates("ffmpeg", "amd")[0])
        self.assertIn("libx264", _video_encode_arg_candidates("ffmpeg", "cpu")[0])


if __name__ == "__main__":
    unittest.main()
