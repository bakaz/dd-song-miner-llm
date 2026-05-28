from __future__ import annotations

import unittest

from dd_song_miner_llm.asr import _is_cuda_runtime_error


class ASRFallbackTests(unittest.TestCase):
    def test_detects_missing_cublas_runtime(self) -> None:
        exc = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

        self.assertTrue(_is_cuda_runtime_error(exc))

    def test_ignores_non_cuda_runtime_errors(self) -> None:
        exc = RuntimeError("audio file is empty")

        self.assertFalse(_is_cuda_runtime_error(exc))


if __name__ == "__main__":
    unittest.main()
