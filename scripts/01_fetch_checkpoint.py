#!/usr/bin/env python3
"""
Download a full model snapshot (a specific training step / revision) from Hugging Face Hub.

Usage examples:
  python scripts/01_fetch_checkpoint.py \
      --model eleutherai/pythia-6.9b-deduped \
      --rev step080000 \
      --outdir /workspace/checkpoints

Notes:
- If the repo is gated or private, run `huggingface-cli login` first on the pod.
- We disable symlinks so the snapshot is a real, portable directory.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download, hf_api
from huggingface_hub.utils import HfHubHTTPError


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def _bytes_to_gb(n: int) -> float:
    return n / (1024 ** 3)


def _free_space_gb(path: Path) -> float:
    st = os.statvfs(str(path))
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def download_snapshot(model: str, revision: str, outdir: Path, force: bool = False, 
                     progress: bool = False, dry_run: bool = False) -> Optional[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / f"{_safe_name(model)}-{revision}"

    print(f"[INFO] Model:    {model}")
    print(f"[INFO] Revision: {revision}")
    print(f"[INFO] Target:   {target}")

    # Quick existence check (idempotent)
    if target.exists() and any(target.iterdir()) and not force:
        print("[INFO] Target directory already exists and is non-empty. Skipping download.")
        print("[INFO] Use --force to re-download.")
        return target

    # Optional: estimate size & free space
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=model, revision=revision)
        # We can't know exact LFS sizes without extra calls; just warn about free space.
        free_gb = _free_space_gb(outdir)
        print(f"[INFO] Free space under {outdir}: ~{free_gb:.1f} GB")
        if free_gb < 50:
            print("[WARN] Less than ~50GB free. Large checkpoints may fail due to space.")
    except Exception as e:
        print(f"[WARN] Could not query repo files/space hint: {e}")

    if dry_run:
        print("[DRY-RUN] Would download to:", target)
        print("[DRY-RUN] Use --force to actually download.")
        return None

    try:
        local_dir = snapshot_download(
            repo_id=model,
            revision=revision,
            local_dir=str(target),
            local_dir_use_symlinks=False,   # real files (safer to move/copy)
            ignore_patterns=["*.md", "*.txt", "LICENSE*"],  # optional: skip docs
            tqdm_class=None if not progress else None  # enable progress bar if requested
        )
        print(f"[OK] Downloaded to: {local_dir}")
        return Path(local_dir)
    except HfHubHTTPError as e:
        print(f"[ERROR] Hugging Face HTTP error: {e}")
        print("        Check the model ID, revision name, and whether you need `huggingface-cli login`.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error during snapshot_download: {e}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Download a model checkpoint (snapshot) from Hugging Face Hub.")
    p.add_argument("--model", required=True, help="e.g. eleutherai/pythia-6.9b-deduped")
    p.add_argument("--rev", required=True, help="e.g. step080000")
    p.add_argument("--outdir", default="checkpoints", help="Where to place the snapshot directory")
    p.add_argument("--force", action="store_true", help="Re-download even if directory already exists")
    p.add_argument("--progress", action="store_true", help="Show download progress bar")
    p.add_argument("--dry-run", action="store_true", help="Check requirements without downloading")
    args = p.parse_args()

    outdir = Path(args.outdir).resolve()
    target = download_snapshot(args.model, args.rev, outdir, args.force, args.progress, args.dry_run)
    
    if target:
        print(f"[DONE] Snapshot ready at: {target}")
    elif args.dry_run:
        print("[DONE] Dry run completed. Use --force to actually download.")


if __name__ == "__main__":
    main()
