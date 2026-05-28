from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import TranscriptSegment


class Transcriber:
    def __init__(self, config: dict[str, Any]) -> None:
        self.settings = config["asr"]
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper not installed. pip install faster-whisper") from exc

        model_name = str(self.settings.get("model", "small"))
        device = str(self.settings.get("device", "auto"))
        compute_type = str(self.settings.get("compute_type", "default"))
        kwargs: dict[str, Any] = {"device": device}
        if compute_type != "default":
            kwargs["compute_type"] = compute_type
        self._model = WhisperModel(model_name, **kwargs)
        return self._model

    def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        model = self._load_model()
        segments, _info = model.transcribe(
            str(audio_path),
            language=self.settings.get("language"),
            beam_size=int(self.settings.get("beam_size", 5)),
            vad_filter=bool(self.settings.get("vad_filter", True)),
            initial_prompt=self.settings.get("initial_prompt"),
        )
        results: list[TranscriptSegment] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                results.append(TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                ))
        return results
