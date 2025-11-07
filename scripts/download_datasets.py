"""Download VLCS and PACS datasets using DomainBed utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from domainbed.scripts import download as db_download


AVAILABLE = {
    "VLCS": db_download.download_vlcs,
    "PACS": db_download.download_pacs,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DomainBed datasets.")
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
        choices=AVAILABLE.keys(),
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
        target = root / name
        if target.exists() and not args.force:
            print(f"[skip] {name} already present at {target}")
            continue
        print(f"[download] {name} -> {target}")
        AVAILABLE[name](str(root))
        print(f"[done] {name}")


if __name__ == "__main__":
    main()
