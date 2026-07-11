#!/usr/bin/env python3
"""Read every regular file below a directory to refresh its access time.

Many shared/HPC filesystems use atime (access time) as an input to their
purge policy.  This program deliberately performs a small *data* read from
each regular file and closes it.  It does not modify, rename, or delete any
file.  It does not follow symbolic links.

Whether a read updates atime ultimately depends on the filesystem's mount and
purge-policy settings.  In particular, a filesystem mounted with ``noatime``
will not record these accesses; confirm the local policy with the storage
administrator before relying on this as retention protection.
"""

from __future__ import annotations

import argparse
import errno
import os
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path


def iter_regular_files(root: Path) -> Iterator[Path]:
    """Yield regular files recursively, without following symlinks."""
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            yield Path(entry.path)
                    except OSError as exc:
                        print(f"WARNING: cannot inspect {entry.path}: {exc}", file=sys.stderr)
        except OSError as exc:
            print(f"WARNING: cannot list {directory}: {exc}", file=sys.stderr)


def read_file(path: Path, byte_count: int) -> None:
    """Open, read data, and close one file without changing its contents."""
    # O_NOFOLLOW closes the small race between scandir and open on platforms
    # that support it.  O_NONBLOCK avoids waiting on an unexpected special
    # file; regular files are selected above, then verified again below.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        os.read(fd, byte_count)
    finally:
        os.close(fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory tree to access recursively")
    parser.add_argument(
        "--bytes",
        type=int,
        default=1,
        help="Bytes of content to read from each file (default: 1)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress after this many files; use 0 to disable (default: 10000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be read without opening them",
    )
    args = parser.parse_args()
    if args.bytes < 1:
        parser.error("--bytes must be at least 1")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.expanduser()
    try:
        root_stat = root.stat()
    except OSError as exc:
        print(f"ERROR: cannot access root {root}: {exc}", file=sys.stderr)
        return 2
    if not stat.S_ISDIR(root_stat.st_mode):
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2

    started = time.monotonic()
    seen = succeeded = failed = 0
    action = "would read" if args.dry_run else "read"
    for path in iter_regular_files(root):
        seen += 1
        if args.dry_run:
            print(path)
            succeeded += 1
            continue
        try:
            read_file(path, args.bytes)
            succeeded += 1
        except OSError as exc:
            failed += 1
            print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)

        if args.progress_every and seen % args.progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"Progress: {seen:,} files scanned, {succeeded:,} {action}, "
                f"{failed:,} failed ({elapsed:.0f}s elapsed)",
                file=sys.stderr,
            )

    elapsed = time.monotonic() - started
    print(
        f"Completed: {seen:,} files scanned, {succeeded:,} {action}, "
        f"{failed:,} failed ({elapsed:.0f}s elapsed)",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
