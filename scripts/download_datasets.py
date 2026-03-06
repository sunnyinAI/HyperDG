"""Download datasets used by this project (DomainBed + WILDS)."""

from __future__ import annotations

import argparse
from pathlib import Path

def _download_vlcs(root: str) -> None:
    from domainbed.scripts import download as db_download

    db_download.download_vlcs(root)


def _download_pacs(root: str) -> None:
    from domainbed.scripts import download as db_download

    db_download.download_pacs(root)


def _download_wilds_camelyon(root: str) -> None:
    from wilds import get_dataset

    get_dataset(dataset="camelyon17", root_dir=root, download=True)


def _download_wilds_fmow(root: str) -> None:
    from wilds import get_dataset

    get_dataset(dataset="fmow", root_dir=root, download=True)


AVAILABLE = {
    "VLCS": {"fn": _download_vlcs, "target": "VLCS"},
    "PACS": {"fn": _download_pacs, "target": "PACS"},
    "WILDSCamelyon": {"fn": _download_wilds_camelyon, "target": "camelyon17_v1.0"},
    "WILDSFMoW": {"fn": _download_wilds_fmow, "target": "fmow_v1.1"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project datasets.")
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Directory to store datasets (default: ./data)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=sorted(AVAILABLE.keys()),
        default=list(AVAILABLE.keys()),
        help="Datasets to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if dataset folder already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        spec = AVAILABLE[name]
        target = root / spec["target"]
        if target.exists() and not args.force:
            print(f"[skip] {name} already present at {target}")
            continue
        print(f"[download] {name} -> {target}")
        spec["fn"](str(root))
        print(f"[done] {name}")


if __name__ == "__main__":
    main()
