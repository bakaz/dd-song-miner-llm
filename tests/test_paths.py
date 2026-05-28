from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dd_song_miner_llm.paths import safe_path_part, should_stage_for_ffmpeg, stage_input_for_ffmpeg


class PathHandlingTests(unittest.TestCase):
    def test_non_ascii_paths_are_staged_to_ascii_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "綾音.mp4"
            source.write_bytes(b"video")

            staged = stage_input_for_ffmpeg(source, root / "stage")

            self.assertTrue(staged.exists())
            self.assertTrue(staged.name.isascii())
            self.assertEqual(staged.read_bytes(), b"video")

    def test_ascii_paths_do_not_need_staging(self) -> None:
        path = Path("input.mp4")

        self.assertFalse(should_stage_for_ffmpeg(path))

    def test_safe_path_part_removes_windows_invalid_chars(self) -> None:
        self.assertEqual(safe_path_part('a:b*c?"d'), "a_b_c__d")


if __name__ == "__main__":
    unittest.main()
