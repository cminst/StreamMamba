#!/usr/bin/env python3
"""
loc.py – Count every line of every *.py file that Git knows about.
Run from the repository root.
"""

import subprocess
import os
from pathlib import Path

def git_tracked_files():
    """Return a list of relative paths that are tracked by Git."""
    cmd = ["git", "ls-files", "-z"]
    out = subprocess.check_output(cmd).decode("utf-8")
    return [p for p in out.split("\0") if p]

def count_lines(path):
    """Return the number of physical lines in the file at *path*."""
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0

def main():
    total_lines = 0
    files_seen = []

    for rel_path in git_tracked_files():
        if not rel_path.endswith(".py"):
            continue
        full_path = Path(rel_path)
        if not full_path.exists():
            continue   # deleted but still in index, ignore

        lines = count_lines(full_path)
        total_lines += lines
        files_seen.append((rel_path, lines))

    # Summary
    print("Lines of code in tracked *.py files\n")
    for path, lines in sorted(files_seen):
        print(f"{lines:>8}  {path}")
    print("-" * 40)
    print(f"{'TOTAL':>8}  {total_lines}")

if __name__ == "__main__":
    main()
