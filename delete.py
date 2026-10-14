"""
cleanup.py
Removes all generated outputs from previous runs.
Run this before starting a fresh pipeline.
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# Directories whose *contents* will be removed (folders themselves are kept)
CLEAN_CONTENTS = [
    "data/processed",
    "models",
    "results",
    "logs",
]

# Directories removed entirely, then recreated empty
RECREATE = [
    "data/processed/frames",
    "data/processed/patient_cache",
    "data/processed/vlm_cache",
    "models/checkpoints",
    "models/vlm",
]


def confirm() -> bool:
    print("=" * 70)
    print("AngioCAD Cleanup Utility")
    print("=" * 70)
    print("This will DELETE all processed data, trained models, and results.")
    print("Raw data (data/raw/) will NOT be touched.")
    print()
    answer = input("Continue? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def remove_path(path: Path) -> None:
    if not path.exists():
        print(f"  skip      {path.relative_to(PROJECT_ROOT)}")
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"  removed   {path.relative_to(PROJECT_ROOT)}/")
    else:
        path.unlink()
        print(f"  removed   {path.relative_to(PROJECT_ROOT)}")


def clean_directory_contents(folder: Path) -> None:
    """Remove everything inside a folder but keep the folder itself."""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  created   {folder.relative_to(PROJECT_ROOT)}/")
        return
    for item in folder.iterdir():
        remove_path(item)


def main() -> int:
    if not confirm():
        print("Aborted.")
        return 1

    print()
    print("Cleaning generated directories...")
    for rel in CLEAN_CONTENTS:
        target = PROJECT_ROOT / rel
        print(f"\n[{rel}]")
        if target.exists():
            clean_directory_contents(target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            print(f"  created   {rel}/")

    print()
    print("Creating empty working directories...")
    for rel in RECREATE:
        target = PROJECT_ROOT / rel
        target.mkdir(parents=True, exist_ok=True)
        print(f"  ready     {rel}/")

    print()
    print("=" * 70)
    print("Cleanup complete. Raw data is untouched.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())