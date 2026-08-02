#!/usr/bin/env python3
"""VLNVerse episode-split downloader for AgentCanvas.

Downloads the episode split files (``raw_data/final_splits/{fine,coarse}_*.json.gz``)
from the Hugging Face dataset ``Eyz/VLNVerse_data`` into
``{REPO_ROOT}/data/vlnverse/raw_data`` — where the env_vlnverse nodeset (and the
``data/vlnverse/raw_data`` symlink laid down by install_ac_vlnverse.sh) expects
them. If you already have a local episode store (e.g. a NavHarness / VLNVerse
checkout), prefer the symlink; this downloader is for machines without one.

This is the episode sibling of ``fetch_scenes_vlnverse.py``: the two datasets
live in separate HF repos (episodes = ``Eyz/VLNVerse_data``, scenes =
``Eyz/VLNVerse_scene``) and share the same ``$VLNVERSE_DATA_ROOT`` contract.
Episodes are 8 small ``.json.gz`` files (~a few MB total), so this needs none
of the scene downloader's rate-limit machinery — one ``snapshot_download`` with
an allow-pattern pulls exactly the splits and nothing else.

The upstream repo also carries a large ``traj_data/`` tree (per-episode
trajectory recordings, ~47k files) that the env nodeset does NOT use — the
allow-pattern deliberately excludes it. Note also that the HF repo ships only
the four canonical splits per dataset (train / val / val_unseen / test); other
splits the panel can enumerate (challenge / human / subset / mini …) are extras
that only exist in a NavHarness checkout, not here.

Examples:
  python3 scripts/data/fetch_episodes_vlnverse.py            # all 8 splits
  python3 scripts/data/fetch_episodes_vlnverse.py --list
  python3 scripts/data/fetch_episodes_vlnverse.py --output-dir /tmp/rd

Exit codes: 0 = done, 1 = error.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Prefer the plain HTTP downloader over the Xet backend (brotli decoder errors
# on some networks); mirrors fetch_scenes_vlnverse.py.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "Eyz/VLNVerse_data"
# The subtree we pull; everything under it, nothing else (excludes traj_data/).
SPLITS_SUBDIR = "raw_data/final_splits"
ALLOW_PATTERN = f"{SPLITS_SUBDIR}/*"

# Single env-var contract with the installer + nodeset runtime, symmetric with
# fetch_scenes_vlnverse.py's VLNVERSE_SCENE_DIR: VLNVERSE_RAW_DATA_DIR wins;
# else derive from VLNVERSE_DATA_ROOT (the dir holding raw_data/ + scene/, the
# same variable install_ac_vlnverse.sh and env_vlnverse's _data_root() read);
# else the repo-local data/vlnverse/raw_data. The derivation keeps a lone
# VLNVERSE_DATA_ROOT export from making this script write a REAL repo-local
# raw_data dir that the runtime never reads and that blocks the installer's
# symlink step.
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("VLNVERSE_RAW_DATA_DIR")
    or (
        Path(os.environ["VLNVERSE_DATA_ROOT"]) / "raw_data"
        if os.environ.get("VLNVERSE_DATA_ROOT")
        else REPO_ROOT / "data" / "vlnverse" / "raw_data"
    )
)


def _hf():
    """Import huggingface_hub lazily so --help/--list-less errors are readable."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install it with: pip install -U huggingface_hub\n"
            "(or run inside the ac-vlnverse conda env)"
        ) from exc
    return HfApi, snapshot_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            f"Download VLNVerse episode splits ({SPLITS_SUBDIR}/*.json.gz) from "
            f"the Hugging Face dataset {DEFAULT_REPO_ID}."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Destination raw_data/ directory (splits land under "
            "<output-dir>/final_splits/). Default: derived from "
            "$VLNVERSE_RAW_DATA_DIR / $VLNVERSE_DATA_ROOT / repo data/vlnverse/raw_data."
        ),
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HF dataset repo id. Default: {DEFAULT_REPO_ID}")
    p.add_argument("--revision", default="main", help="HF repo revision. Default: main")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (default: $HF_TOKEN).")
    p.add_argument("--list", action="store_true", help="List the split files available in the repo and exit.")
    return p.parse_args()


def list_splits(repo_id: str, revision: str, token: str | None) -> int:
    HfApi, _ = _hf()
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision, token=token)
    splits = sorted(f for f in files if f.startswith(SPLITS_SUBDIR + "/"))
    if not splits:
        print(f"No files under {SPLITS_SUBDIR}/ in {repo_id}.", file=sys.stderr)
        return 1
    print(f"{len(splits)} split file(s) in {repo_id}:")
    for f in splits:
        print("  ", f[len(SPLITS_SUBDIR) + 1 :])
    return 0


def main() -> int:
    args = parse_args()
    if args.list:
        return list_splits(args.repo_id, args.revision, args.token)

    _, snapshot_download = _hf()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # The repo lays the splits at raw_data/final_splits/*; local_dir=<out>'s
    # parent would double the raw_data/ segment, so download into out.parent
    # when out is named raw_data, else into a temp layout. Simplest robust form:
    # snapshot into a staging root and let the repo path decide the layout, then
    # the caller's <out> IS the raw_data dir, so we target out.parent as the
    # local_dir and let HF recreate raw_data/final_splits under it.
    local_root = out.parent if out.name == "raw_data" else out
    print(f"Downloading {ALLOW_PATTERN} from {args.repo_id} -> {local_root}/raw_data/final_splits ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        token=args.token,
        local_dir=str(local_root),
        allow_patterns=[ALLOW_PATTERN],
    )

    dest = local_root / SPLITS_SUBDIR
    got = sorted(p.name for p in dest.glob("*.json*")) if dest.is_dir() else []
    if not got:
        print(f"[ERROR] no split files landed under {dest}", file=sys.stderr)
        return 1
    print(f"Done — {len(got)} split file(s) in {dest}:")
    for name in got:
        print("  ", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
