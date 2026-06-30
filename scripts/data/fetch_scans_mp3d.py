#!/usr/bin/env python3
"""Matterport3D scan downloader for AgentCanvas.

Python-3 port of the upstream ``download_mp.py`` (originally Python 2)
with AgentCanvas-specific defaults:

* Default output dir: ``{REPO_ROOT}/data/mp3d``
  (scans land under ``data/mp3d/v1/scans/{scan_id}/*.zip``)
* Default filetype:   ``matterport_skybox_images``
  (what MatterSim needs — see ``workspace/nodesets/server/matterport3d.py``)

The ToU prompt is preserved. Use ``--accept-tos`` to skip it in automation —
you remain legally bound by the MP Terms of Use at
http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf

Examples:
    # All 90 scans, skybox only — the usual case for MatterSim.
    python3 scripts/data/fetch_scans_mp3d.py

    # One scan, skybox + depth.
    python3 scripts/data/fetch_scans_mp3d.py --id 17DRP5sb8fy \\
        --type matterport_skybox_images undistorted_depth_images

    # Automation-friendly (CI, scripted install).
    python3 scripts/data/fetch_scans_mp3d.py --accept-tos
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

BASE_URL = "http://kaldir.vc.in.tum.de/matterport/"
RELEASE = "v1/scans"
TOS_URL = BASE_URL + "MP_TOS.pdf"

FILETYPES = [
    "cameras",
    "matterport_camera_intrinsics",
    "matterport_camera_poses",
    "matterport_color_images",
    "matterport_depth_images",
    "matterport_hdr_images",
    "matterport_mesh",
    "matterport_skybox_images",
    "undistorted_camera_parameters",
    "undistorted_color_images",
    "undistorted_depth_images",
    "undistorted_normal_images",
    "house_segmentations",
    "region_segmentations",
    "image_overlap_data",
    "poisson_meshes",
    "sens",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "mp3d"
DEFAULT_FILETYPE = "matterport_skybox_images"


def get_release_scans(release_url: str) -> list[str]:
    with urlopen(release_url) as resp:
        return [line.decode().rstrip("\n") for line in resp if line.strip()]


def download_file(url: str, out_file: Path) -> None:
    if out_file.is_file():
        print(f"  [skip] exists: {out_file}")
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url} -> {out_file}")
    fh, tmp_path = tempfile.mkstemp(dir=out_file.parent)
    os.close(fh)
    try:
        urlretrieve(url, tmp_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    os.rename(tmp_path, out_file)


def download_scan(scan_id: str, out_dir: Path, file_types: list[str]) -> None:
    print(f"[scan] {scan_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ft in file_types:
        download_file(f"{BASE_URL}{RELEASE}/{scan_id}/{ft}.zip", out_dir / f"{ft}.zip")


def prompt_tos(accept: bool) -> None:
    if accept:
        print(f"[--accept-tos] MP ToU acknowledged: {TOS_URL}")
        return
    print("By continuing you confirm that you have read and agreed to the MP Terms of Use:")
    print(f"  {TOS_URL}")
    print("Press Enter to continue, or CTRL-C to abort.")
    try:
        input("")
    except (KeyboardInterrupt, EOFError):
        sys.exit(130)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Matterport3D scan assets (AgentCanvas Py3 port).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"base dir; scans go to {{out_dir}}/v1/scans/ (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--id",
        default="ALL",
        help="scan id, or ALL for all 90 scans (default: ALL)",
    )
    parser.add_argument(
        "--type",
        nargs="+",
        default=[DEFAULT_FILETYPE],
        choices=FILETYPES,
        metavar="FT",
        help=f"filetypes to fetch (default: {DEFAULT_FILETYPE}); any of:\n  "
        + ", ".join(FILETYPES),
    )
    parser.add_argument(
        "--accept-tos",
        action="store_true",
        help="skip interactive ToU prompt (you remain legally bound by the ToU)",
    )
    args = parser.parse_args()

    prompt_tos(args.accept_tos)

    release_scans = get_release_scans(f"{BASE_URL}{RELEASE}.txt")
    base_dir = Path(args.out_dir).resolve()
    scans_root = base_dir / RELEASE

    if args.id == "ALL":
        print(
            f"Downloading {len(release_scans)} scans x {len(args.type)} filetype(s) -> {scans_root}"
        )
        for scan_id in release_scans:
            download_scan(scan_id, scans_root / scan_id, args.type)
    else:
        if args.id not in release_scans:
            print(f"ERROR: invalid scan id: {args.id}", file=sys.stderr)
            return 1
        download_scan(args.id, scans_root / args.id, args.type)

    print(f"\nDone. Scans under: {scans_root}")
    print("MatterSim reads the *.zip directly — no extraction needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
