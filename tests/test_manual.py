from __future__ import annotations

import unittest

from dd_song_miner_llm.manual import _parse_time


class ManualCutTests(unittest.TestCase):
    def test_parse_time_accepts_timecode_and_seconds(self) -> None:
        self.assertEqual(_parse_time("01:02:03"), 3723.0)
        self.assertEqual(_parse_time("02:03"), 123.0)
        self.assertEqual(_parse_time("12.5"), 12.5)


if __name__ == "__main__":
    unittest.main()
