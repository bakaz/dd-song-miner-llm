from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dd_song_miner_llm.models import SongMatch, TranscriptSegment
from dd_song_miner_llm.report import write_match_context_reports


class MatchContextReportTests(unittest.TestCase):
    def test_match_context_includes_neighboring_segments(self) -> None:
        segments = [TranscriptSegment(float(i), float(i + 1), f"text {i}") for i in range(8)]
        matches = [SongMatch("Song", "Artist", "snippet", [3, 4], 0.9)]

        with tempfile.TemporaryDirectory() as tmp:
            _csv_path, json_path = write_match_context_reports(matches, segments, Path(tmp), context_segments=2)
            data = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(data[0]["segment_indices"], [3, 4])
        self.assertEqual([item["segment_index"] for item in data[0]["context_segments"]], [1, 2, 3, 4, 5, 6])
        self.assertEqual([item["segment_index"] for item in data[0]["matched_segments"]], [3, 4])


if __name__ == "__main__":
    unittest.main()
