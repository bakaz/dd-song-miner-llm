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
        completed_videos = _load_marker(marker)

        folder_runs: list[dict[str, Any]] = []
        folder_ok = True
        has_work = False

        for video in sorted(by_folder[folder]):
            video_key = str(video.resolve())

            if video_key in completed_videos and completed_videos[video_key].get("status") == "success":
                print(f"[skip] Already processed: {video}")
                folder_runs.append(completed_videos[video_key])
                runs.append(completed_videos[video_key])
                continue

            has_work = True
            rel_folder = _relative_folder(root, folder)
            run_name = safe_path_part(video.stem)
            run_dir = work / rel_folder / run_name
            result_dir = results_root / rel_folder / run_name

            print(f"[run] {video}")
            try:
                song_results = run_pipeline(video, run_dir, config)
                if run_dir.resolve() != result_dir.resolve():
                    shutil.copytree(run_dir, result_dir, dirs_exist_ok=True)
                item = {
                    "video": str(video),
                    "video_key": video_key,
                    "work_dir": str(run_dir),
                    "result_dir": str(result_dir),
                    "song_count": len(song_results),
                    "status": "success",
                }
                folder_runs.append(item)
                runs.append(item)
                completed_videos[video_key] = item
            except Exception as exc:
                folder_ok = False
                item = {
                    "video": str(video),
                    "video_key": video_key,
                    "work_dir": str(run_dir),
                    "result_dir": str(result_dir),
                    "error": str(exc),
                    "status": "failed",
                }
                folder_runs.append(item)
                runs.append(item)
                completed_videos[video_key] = item
                print(f"[error] {video}: {exc}")

            _write_marker(marker, completed_videos)

        if has_work and folder_ok:
            print(f"[done] All videos processed in: {folder}")
        elif has_work:
            print(f"[warn] Some videos failed in: {folder}, will retry on next run")

    return runs


def _relative_folder(root: Path, folder: Path) -> Path:
    try:
        rel = folder.relative_to(root)
    except ValueError:
        rel = Path(safe_path_part(folder.name))
    if str(rel) == ".":
        return Path("_root")
    return Path(*[safe_path_part(part) for part in rel.parts])


def _load_marker(marker: Path) -> dict[str, dict[str, Any]]:
    if not marker.exists():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "videos" in data:
            return data["videos"]
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_marker(marker: Path, completed_videos: dict[str, dict[str, Any]]) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "videos": completed_videos,
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
