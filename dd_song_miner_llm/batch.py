from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import VIDEO_EXTENSIONS, iter_video_files, safe_path_part
from .pipeline import run_pipeline


def run_batch(
    input_root: str | Path,
    result_root: str | Path,
    work_root: str | Path,
    config: dict[str, Any],
    marker_name: str = ".dd_song_miner_done.json",
    extensions: set[str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(input_root).expanduser()
    results_root = Path(result_root)
    work = Path(work_root)
    results_root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    videos = iter_video_files(root, extensions or VIDEO_EXTENSIONS)
    by_folder: dict[Path, list[Path]] = {}
    for video in videos:
        by_folder.setdefault(video.parent, []).append(video)

    runs: list[dict[str, Any]] = []
    for folder in sorted(by_folder):
        marker = folder / marker_name
        if marker.exists():
            print(f"[skip] Marker exists: {marker}")
            continue

        folder_ok = True
        folder_runs: list[dict[str, Any]] = []
        for video in sorted(by_folder[folder]):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rel_folder = _relative_folder(root, folder)
            run_name = f"{safe_path_part(video.stem)}_{stamp}"
            run_dir = work / rel_folder / run_name
            result_dir = results_root / rel_folder / run_name

            print(f"[run] {video}")
            try:
                song_results = run_pipeline(video, run_dir, config)
                if run_dir.resolve() != result_dir.resolve():
                    shutil.copytree(run_dir, result_dir, dirs_exist_ok=True)
                item = {
                    "video": str(video),
                    "work_dir": str(run_dir),
                    "result_dir": str(result_dir),
                    "song_count": len(song_results),
                }
                folder_runs.append(item)
                runs.append(item)
            except Exception as exc:
                folder_ok = False
                item = {
                    "video": str(video),
                    "work_dir": str(run_dir),
                    "result_dir": str(result_dir),
                    "error": str(exc),
                }
                folder_runs.append(item)
                runs.append(item)
                print(f"[error] {video}: {exc}")

        if folder_ok:
            try:
                _write_marker(marker, folder_runs)
            except OSError as exc:
                print(f"[warn] Could not write marker {marker}: {exc}")
        else:
            print(f"[warn] Not writing marker because at least one video failed: {folder}")

    return runs


def _relative_folder(root: Path, folder: Path) -> Path:
    try:
        rel = folder.relative_to(root)
    except ValueError:
        rel = Path(safe_path_part(folder.name))
    if str(rel) == ".":
        return Path("_root")
    return Path(*[safe_path_part(part) for part in rel.parts])


def _write_marker(marker: Path, runs: list[dict[str, Any]]) -> None:
    payload = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
