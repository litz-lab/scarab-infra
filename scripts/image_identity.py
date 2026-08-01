#!/usr/bin/python3
"""Content-addressed docker image identity for scarab-infra.

The image tag used to be the scarab-infra git hash. That made every infra
commit mint a new tag for byte-identical image content: the tag had to be
materialised on each node, either by retagging a local base image or -- when
that base was not present -- by pulling multiple GB from ghcr just to rename
it. It was also simply the wrong key, because since the workflow scripts moved
to a bind mount (see common/Dockerfile.common) most commits cannot change image
content at all.

The tag is now a hash of the repository content that `docker build` bakes into
the image, so an image is rebuilt or fetched if and only if it actually
differs. Identical content yields an identical tag on every node and on every
branch, which means `image_exist()` simply hits and no build, retag or pull
happens at all. See image_content_hash() for what the hash does and does not
cover.

This module deliberately imports nothing outside the standard library so that
CI can call it (`python3 -m scripts.image_identity <workload_group>`) from a
bare checkout, without installing the docker SDK or the rest of the infra deps.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Length of the hex digest used in tags. Long enough that a collision across a
# lab's worth of images is not a practical concern, short enough to read.
TAG_HASH_LEN = 12


def image_context_pathspec(workload_group: str) -> list:
    """Git pathspecs for the repository content baked into a workload's image.

    Included: the Dockerfiles under common/, the workload's own directory, and
    fingerprint_src/ (Dockerfile.common COPYs it in and compiles it into
    libfpg.so).

    Excluded: common/scripts and the per-workload workload_*_entrypoint.sh.
    Those are bind-mounted from the host checkout at run time rather than
    COPY'd in, so editing them cannot change image content and must not force a
    rebuild -- that is the entire point of mounting them.

    This is a hand-maintained approximation of the docker build context, and
    nothing enforces that it stays accurate. If a Dockerfile starts COPYing
    from a path outside workloads/<group>/, add that path here or its changes
    will not be noticed. workloads/oss/Dockerfile already does this (it COPYs
    ./workloads//OSS/setup.sh); that image is unbuildable for other reasons,
    but it is the failure mode to watch for.
    """
    return [
        "common",
        "fingerprint_src",
        f"workloads/{workload_group}",
        ":(exclude)common/scripts",
        f":(exclude)workloads/{workload_group}/workload_root_entrypoint.sh",
        f":(exclude)workloads/{workload_group}/workload_user_entrypoint.sh",
    ]


def _context_files(workload_group: str, infra_dir) -> list:
    """Paths under the image context, tracked plus untracked-but-not-ignored.

    Untracked files have to be included: `docker build` streams the whole
    context, and several Dockerfiles COPY entire directories (e.g. `COPY
    fingerprint_src ...`), so a file that is merely not `git add`ed yet still
    lands in the image. Listing only tracked files would let two developers
    with different untracked files share a tag -- and therefore each other's
    cached image.

    --exclude-standard applies .gitignore, which is why .gitignore carries the
    same junk patterns as .dockerignore (*~, #*#, __pycache__): a file docker
    ignores must be ignored here too, or editor backups would churn the tag.
    """
    pathspec = image_context_pathspec(workload_group)
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--"] + pathspec,
        cwd=str(infra_dir),
        check=True,
        capture_output=True,
    ).stdout
    # A path can be reported twice when it is both cached and modified.
    return sorted({p for p in out.split(b"\0") if p})


def image_content_hash(workload_group: str, infra_dir=None) -> str:
    """Hash the on-disk content of everything baked into the workload's image.

    Covers the working-tree bytes and the executable bit of every file in the
    image context, tracked or not, so an uncommitted Dockerfile edit, a new
    untracked file, or a `chmod +x` all produce a different tag. A local
    experiment therefore cannot silently reuse an image built from different
    sources.

    Two things it does NOT cover:

    * `image_context_pathspec()` is a hand-maintained description of the build
      context. A Dockerfile that COPYs from outside `workloads/<group>/`
      (workloads/oss/Dockerfile does) falls outside the hash.
    * The build *inputs* are hashed, not the resulting image bits. Unpinned
      fetches -- the DynamoRIO clone, the SimPoint and pmu-tools clones, the
      pin download, `FROM ubuntu:focal`, `apt install` -- mean two builds of
      one tag can still differ. Same caveat the git-hash tag had; it only
      matters when a rebuild actually happens.

    Deliberately uncached: the inputs are tiny (tens of files, a few hundred
    KB), so memoising saves nothing while risking a long-lived process holding
    a stale tag across an edit.
    """
    infra_dir = Path(infra_dir) if infra_dir else REPO_ROOT

    digest = hashlib.sha256()
    for rel in _context_files(workload_group, infra_dir):
        path = infra_dir / rel.decode()
        digest.update(rel)
        digest.update(b"\0")
        try:
            # COPY preserves the exec bit, so it is part of the image content.
            digest.update(b"x" if path.stat().st_mode & 0o111 else b"-")
            digest.update(path.read_bytes())
        except (OSError, IsADirectoryError):
            # A listed path missing from the worktree (e.g. sparse checkout) is
            # itself part of the identity; record it rather than ignoring it.
            digest.update(b"<absent>")
        digest.update(b"\0")

    return digest.hexdigest()[:TAG_HASH_LEN]


def image_tag_for(workload_group: str, infra_dir=None) -> str:
    """Full local image reference, e.g. 'spec2017:3f9a12c0be44'."""
    return f"{workload_group}:{image_content_hash(workload_group, infra_dir)}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <workload_group>", file=sys.stderr)
        raise SystemExit(2)
    print(image_content_hash(sys.argv[1]))
