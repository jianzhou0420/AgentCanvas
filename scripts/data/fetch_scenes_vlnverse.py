#!/usr/bin/env python3
"""VLNVerse scene downloader for AgentCanvas.

Downloads kujiale scene folders (USD + Materials/Meshes + freemap.npy +
room_region.json) from the Hugging Face dataset ``Eyz/VLNVerse_scene`` into
``{REPO_ROOT}/data/vlnverse/scene`` — where the env_vlnverse nodeset (and the
``data/vlnverse/scene`` symlink laid down by install_ac_vlnverse.sh) expects
them. If you already have a local scene store (e.g. a NavHarness checkout),
prefer the symlink; this downloader is for machines without one.

Adapted from ``/home/xunyi/Desktop/Data/script/download_vlnverse_scene.py``
(same rate-limit barrier, per-scene ``.download_complete`` markers, and exit
codes); AgentCanvas changes: repo-root-relative default output dir
(``$VLNVERSE_SCENE_DIR`` override), and huggingface_hub is imported lazily so
``--help`` works without the dependency.

Examples:
  python3 scripts/data/fetch_scenes_vlnverse.py --scene kujiale_0011
  python3 scripts/data/fetch_scenes_vlnverse.py kujiale_0011 kujiale_0005
  python3 scripts/data/fetch_scenes_vlnverse.py --list-scenes
  python3 scripts/data/fetch_scenes_vlnverse.py

When no scene is specified, all scenes that are not already present are
downloaded. Exit codes: 0 = done, 42 = rate-limited with
--exit-on-rate-limit (see fetch_scenes_vlnverse_loop.sh), 1 = error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# The Xet backend can fail on some networks/environments with brotli decoder
# errors. Prefer the regular HTTP downloader unless the user opts back in.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Silence per-file byte-transfer bars from hf_hub_download (called internally
# by snapshot_download). Snapshot_download's aggregate "Fetching N files" bar
# bypasses this because we hand in our own tqdm_class.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm ships with huggingface_hub in most envs
    tqdm = None
else:
    # Required for safe multi-bar rendering across worker threads.
    tqdm.set_lock(threading.RLock())


def _hf():
    """Import huggingface_hub lazily so --help works without the dependency."""
    try:
        import huggingface_hub as hf
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install it with: pip install -U huggingface_hub\n"
            "(or use the ac-vlnverse env: bash scripts/install/install_ac_vlnverse.sh)"
        ) from exc
    return hf


def _log(message: str) -> None:
    """Print without scribbling over active tqdm bars."""
    if tqdm:
        tqdm.write(message)
    else:
        print(message)


# Global rate-limit barrier: when any worker sees HTTP 429, it sets a shared
# "unblocked-at" timestamp from the Retry-After hint. All workers consult this
# before issuing a new request, so we wait out the window once instead of a
# thundering herd of threads each tripping it again.
_rate_limit_lock = threading.Lock()
_rate_limited_until = 0.0
_RETRY_AFTER_RE = re.compile(r"[Rr]etry[- ]?[Aa]fter[: ]*(\d+)")


def _wait_for_rate_limit() -> None:
    while True:
        with _rate_limit_lock:
            wait = _rate_limited_until - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 5.0))


def _trigger_rate_limit(seconds: float) -> None:
    global _rate_limited_until
    until = time.time() + max(seconds, 0.0)
    with _rate_limit_lock:
        already = _rate_limited_until > time.time()
        if until > _rate_limited_until:
            _rate_limited_until = until
    if not already:
        _log(f"[rate-limit] HF returned 429; pausing all workers for ~{int(seconds)}s")


class _RateLimitedAbort(Exception):
    """Signal: HF returned 429 and --exit-on-rate-limit is set. Bubble up to main."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate-limited; retry-after={retry_after:.0f}s")
        self.retry_after = retry_after


def _retry_after_from(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            header = response.headers.get("Retry-After")
        except Exception:  # noqa: BLE001
            header = None
        if header:
            try:
                return float(header)
            except (TypeError, ValueError):
                pass
        try:
            if getattr(response, "status_code", None) != 429:
                return None
        except Exception:  # noqa: BLE001
            pass
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return float(match.group(1))
    return None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "Eyz/VLNVerse_scene"
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("VLNVERSE_SCENE_DIR", REPO_ROOT / "data" / "vlnverse" / "scene")
)
COMPLETE_MARKER = ".download_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected scene folders from the Hugging Face dataset "
            f"{DEFAULT_REPO_ID}. If no scene is provided, download all missing scenes."
        )
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        help="Scene folder names to download, for example: kujiale_0011",
    )
    parser.add_argument(
        "--scene",
        dest="scene_options",
        action="append",
        default=[],
        help=(
            "Scene folder name to download. Can be repeated and also accepts "
            "comma-separated values."
        ),
    )
    parser.add_argument(
        "--scene-file",
        type=Path,
        help="Text file containing one scene name per line. Lines beginning with # are ignored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Download directory. Default: data/vlnverse/scene under the repo "
            "root ($VLNVERSE_SCENE_DIR overrides)."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision, branch, tag, or commit. Default: main",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN if set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel file download workers within a scene. Default: 8",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries per file before failing the scene. Default: 5",
    )
    parser.add_argument(
        "--exit-on-rate-limit",
        action="store_true",
        help=(
            "Abort the process on the first HTTP 429 with exit code 42, instead "
            "of waiting out the Retry-After window. Intended for wrapper loops "
            "that schedule retries between rate-limit windows."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List available top-level scene folders and exit.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip checking whether requested scenes exist before downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download requested scenes even if their local folders already exist.",
    )
    parser.add_argument(
        "--trust-existing",
        action="store_true",
        help=(
            "Skip non-empty scene folders even if they do not contain the "
            "completion marker. By default, only .download_complete is treated "
            "as already downloaded."
        ),
    )
    return parser.parse_args()


def split_scene_values(values: Iterable[str]) -> list[str]:
    scenes: list[str] = []
    for value in values:
        for part in value.split(","):
            scene = part.strip().strip("/")
            if scene:
                scenes.append(scene)
    return scenes


def read_scene_file(path: Path) -> list[str]:
    scenes: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if "," in value:
            raise ValueError(
                f"{path}:{lineno}: scene-file expects one scene per line, got comma-separated text"
            )
        scenes.append(value.strip("/"))
    return scenes


def collect_requested_scenes(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    requested.extend(split_scene_values(args.scene_options))
    requested.extend(split_scene_values(args.scenes))
    if args.scene_file:
        requested.extend(read_scene_file(args.scene_file))

    seen: set[str] = set()
    unique: list[str] = []
    for scene in requested:
        if "/" in scene or "\\" in scene:
            raise ValueError(
                f"Scene names must be top-level folder names, got: {scene!r}"
            )
        if scene not in seen:
            seen.add(scene)
            unique.append(scene)
    return unique


def list_available_scenes(api, repo_id: str, revision: str, token: str | None) -> list[str]:
    scenes: list[str] = []
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=False,
        token=token,
    ):
        if item.__class__.__name__ == "RepoFolder":
            scenes.append(item.path.rstrip("/"))
    return sorted(scenes)


def validate_scenes(requested: list[str], available: list[str]) -> None:
    available_set = set(available)
    missing = [scene for scene in requested if scene not in available_set]
    if not missing:
        return

    preview = "\n".join(f"  - {scene}" for scene in available[:30])
    extra = "" if len(available) <= 30 else f"\n  ... and {len(available) - 30} more"
    raise ValueError(
        "Requested scene(s) not found: "
        + ", ".join(missing)
        + "\nAvailable scene examples:\n"
        + preview
        + extra
    )


def list_scene_files(
    api,
    repo_id: str,
    revision: str,
    token: str | None,
    scene: str,
) -> list[str]:
    files: list[str] = []
    for item in api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=scene,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        token=token,
    ):
        if item.__class__.__name__ == "RepoFile":
            files.append(item.path)
    if not files:
        raise ValueError(f"No files found under scene folder: {scene}")
    return sorted(files)


def scene_dir(output_dir: Path, scene: str) -> Path:
    return output_dir / scene


def scene_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.name != COMPLETE_MARKER for child in path.iterdir())


def is_scene_downloaded(output_dir: Path, scene: str, trust_existing: bool) -> bool:
    path = scene_dir(output_dir, scene)
    if (path / COMPLETE_MARKER).is_file():
        return True
    return trust_existing and scene_has_files(path)


def mark_scene_complete(output_dir: Path, scene: str, repo_id: str, revision: str) -> None:
    path = scene_dir(output_dir, scene)
    if not path.is_dir():
        return

    marker = path / COMPLETE_MARKER
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    marker.write_text(
        f"repo_id={repo_id}\nrevision={revision}\ncompleted_at={timestamp}\n",
        encoding="utf-8",
    )


def filter_download_targets(
    output_dir: Path,
    scenes: list[str],
    force: bool,
    trust_existing: bool,
) -> tuple[list[str], list[str]]:
    if force:
        return scenes, []

    skipped: list[str] = []
    targets: list[str] = []
    for scene in scenes:
        if is_scene_downloaded(output_dir, scene, trust_existing):
            skipped.append(scene)
        else:
            targets.append(scene)
    return targets, skipped


def download_one_file(
    repo_id: str,
    revision: str,
    output_dir: Path,
    path_in_repo: str,
    args: argparse.Namespace,
) -> str:
    hf = _hf()
    last_error: Exception | None = None
    retries = max(1, args.retries)
    # Rate-limit waits are not counted as failed attempts; bound them separately
    # so a persistent throttle still terminates eventually.
    rate_limit_retries = 0
    max_rate_limit_retries = max(10, retries * 2)
    attempt = 0
    while True:
        _wait_for_rate_limit()
        try:
            return hf.hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                revision=revision,
                local_dir=str(output_dir),
                token=args.token,
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                force_download=args.force,
                headers={"Accept-Encoding": "identity"},
            )
        except Exception as exc:  # noqa: BLE001 - retry all transient download failures
            last_error = exc
            retry_after = _retry_after_from(exc)
            if retry_after is not None:
                if args.exit_on_rate_limit:
                    raise _RateLimitedAbort(retry_after) from exc
                rate_limit_retries += 1
                if rate_limit_retries > max_rate_limit_retries:
                    break
                _trigger_rate_limit(retry_after + 1.0)
                continue
            attempt += 1
            if attempt >= retries:
                break
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"{path_in_repo}: {last_error}")


def download_files(
    repo_id: str,
    revision: str,
    output_dir: Path,
    files: list[str],
    args: argparse.Namespace,
    desc: str,
) -> None:
    max_workers = max(1, args.max_workers)
    failures: list[str] = []

    def record_failure(path: str, exc: BaseException) -> None:
        failures.append(f"{path}: {exc}")

    if max_workers == 1:
        progress = tqdm(files, desc=desc) if tqdm else files
        for path in progress:
            try:
                download_one_file(repo_id, revision, output_dir, path, args)
            except _RateLimitedAbort:
                raise
            except Exception as exc:  # noqa: BLE001 - report every failed file
                record_failure(path, exc)
        if failures:
            raise RuntimeError("\n".join(failures[:20]))
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                download_one_file,
                repo_id,
                revision,
                output_dir,
                path,
                args,
            ): path
            for path in files
        }
        futures = as_completed(future_to_path)
        progress = (
            tqdm(futures, total=len(files), desc=desc) if tqdm else futures
        )
        try:
            for future in progress:
                path = future_to_path[future]
                try:
                    future.result()
                except _RateLimitedAbort:
                    raise
                except Exception as exc:  # noqa: BLE001 - report every failed file
                    record_failure(path, exc)
        except _RateLimitedAbort:
            # Cancel pending tasks for a fast exit; running tasks finish on their own
            # (they're tiny — already-in-flight files complete in a few seconds).
            for f in future_to_path:
                f.cancel()
            raise

    if failures:
        shown = "\n".join(failures[:20])
        hidden = "" if len(failures) <= 20 else f"\n... and {len(failures) - 20} more"
        raise RuntimeError(f"{len(failures)} file(s) failed:\n{shown}{hidden}")


def download_scene(
    api,
    repo_id: str,
    revision: str,
    output_dir: Path,
    scene: str,
    args: argparse.Namespace,
) -> str | None:
    try:
        files = list_scene_files(
            api,
            repo_id=repo_id,
            revision=revision,
            token=args.token,
            scene=scene,
        )
    except Exception as exc:  # noqa: BLE001 - listing failure → fail this scene only
        return f"list_scene_files: {exc}"

    _log(f"{scene}: {len(files)} file(s)")
    try:
        download_files(repo_id, revision, output_dir, files, args, desc=scene)
    except _RateLimitedAbort:
        raise
    except Exception as exc:  # noqa: BLE001 - aggregate per-scene failures
        return str(exc)

    mark_scene_complete(output_dir, scene, repo_id, revision)
    return None


def download(
    api,
    repo_id: str,
    revision: str,
    output_dir: Path,
    scenes: list[str],
    args: argparse.Namespace,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(scenes)} scene(s) into {output_dir}:")
    for scene in scenes:
        print(f"  - {scene}")

    failures: list[str] = []
    for scene in scenes:
        error = download_scene(api, repo_id, revision, output_dir, scene, args)
        if error:
            failures.append(f"{scene}: {error}")

    if failures:
        shown = "\n".join(failures[:20])
        hidden = "" if len(failures) <= 20 else f"\n... and {len(failures) - 20} more"
        raise RuntimeError(f"{len(failures)} scene(s) failed:\n{shown}{hidden}")

    return str(output_dir)


def main() -> int:
    args = parse_args()
    api = _hf().HfApi(token=args.token)

    try:
        requested = collect_requested_scenes(args)

        if args.list_scenes:
            available = list_available_scenes(api, args.repo_id, args.revision, args.token)
            print("\n".join(available))
            print(f"\nTotal scenes: {len(available)}")
            return 0

        available: list[str] | None = None
        if requested and not args.skip_validation:
            available = list_available_scenes(api, args.repo_id, args.revision, args.token)
            validate_scenes(requested, available)

        if requested:
            scenes = requested
        else:
            available = available or list_available_scenes(
                api, args.repo_id, args.revision, args.token
            )
            scenes = available
            print(f"No scene specified; checking all {len(scenes)} scenes.")

        targets, skipped = filter_download_targets(
            args.output_dir,
            scenes,
            force=args.force,
            trust_existing=args.trust_existing,
        )

        if skipped:
            print(f"Skipping {len(skipped)} existing scene(s):")
            for scene in skipped:
                print(f"  - {scene}")

        if not targets:
            print("Nothing to download.")
            return 0

        local_path = download(api, args.repo_id, args.revision, args.output_dir, targets, args)
    except _RateLimitedAbort as exc:
        print(
            f"[rate-limit] HF returned 429 (retry-after={int(exc.retry_after)}s); "
            "exiting with code 42 so a wrapper loop can schedule the next attempt.",
            file=sys.stderr,
        )
        return 42
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Files are available under: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
