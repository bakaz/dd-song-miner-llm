from __future__ import annotations

import unittest

from dd_song_miner_llm.merger import build_song_results
from dd_song_miner_llm.models import SongMatch, TranscriptSegment


def _config(before: float = 5.0, after: float = 5.0) -> dict:
    return {
        "padding": {
            "before_seconds": before,
            "after_seconds": after,
            "min_song_seconds": 0.0,
            "merge_gap_seconds": 30.0,
        }
    }


class PaddingTests(unittest.TestCase):
    def test_padding_is_clamped_to_neighboring_segments(self) -> None:
        segments = [
            TranscriptSegment(0.0, 8.0, "talk before"),
            TranscriptSegment(10.0, 20.0, "lyric one"),
            TranscriptSegment(20.0, 30.0, "lyric two"),
            TranscriptSegment(31.0, 40.0, "talk after"),
        ]
        matches = [SongMatch("Song", "", "", [1, 2], 0.9)]

        results = build_song_results(segments, matches, 40.0, _config())

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].start, 8.0)
        self.assertEqual(results[0].end, 31.0)

    def test_padding_uses_media_bounds_without_neighbors(self) -> None:
        segments = [TranscriptSegment(2.0, 10.0, "only song")]
        matches = [SongMatch("Song", "", "", [0], 0.9)]

        results = build_song_results(segments, matches, 12.0, _config())

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].start, 0.0)
        self.assertEqual(results[0].end, 12.0)

    def test_overlapping_neighbor_timestamps_do_not_trim_song(self) -> None:
        segments = [
            TranscriptSegment(0.0, 12.0, "overlap before"),
            TranscriptSegment(10.0, 30.0, "song"),
            TranscriptSegment(25.0, 40.0, "overlap after"),
        ]
        matches = [SongMatch("Song", "", "", [1], 0.9)]

        results = build_song_results(segments, matches, 40.0, _config())

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].start, 10.0)
        self.assertEqual(results[0].end, 30.0)

    def test_merged_song_padding_uses_last_merged_segment(self) -> None:
        segments = [
            TranscriptSegment(0.0, 5.0, "talk before"),
            TranscriptSegment(10.0, 20.0, "song first"),
            TranscriptSegment(25.0, 35.0, "song second"),
            TranscriptSegment(40.0, 45.0, "talk after"),
        ]
        matches = [
            SongMatch("Song", "", "", [1], 0.8),
            SongMatch("Song", "", "", [2], 0.9),
        ]

        results = build_song_results(segments, matches, 45.0, _config(before=5.0, after=10.0))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].start, 5.0)
        self.assertEqual(results[0].end, 40.0)


if __name__ == "__main__":
    unittest.main()
