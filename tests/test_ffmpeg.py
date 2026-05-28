from __future__ import annotations

import unittest

from dd_song_miner_llm.ffmpeg import _parse_ffmpeg_duration


class FFmpegDurationTests(unittest.TestCase):
    def test_parse_duration_from_ffmpeg_stderr(self) -> None:
        text = "Input #0, mp4, from 'video.mp4':\n  Duration: 01:02:03.45, start: 0.000000"

        self.assertAlmostEqual(_parse_ffmpeg_duration(text), 3723.45)

    def test_parse_duration_returns_none_when_missing(self) -> None:
        self.assertIsNone(_parse_ffmpeg_duration("no duration here"))


if __name__ == "__main__":
    unittest.main()
