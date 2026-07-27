#!/usr/bin/python3
"""Content-addressed docker image identity for scarab-infra.

The image tag used to be the scarab-infra git hash. That made every infra
commit mint a new tag for byte-identical image content: the tag had to be
materialised on each node, either by retagging a local base image or -- when
that base was not present -- by pulling multiple GB from ghcr just to rename
it. It was also simply the wrong key, because since the workflow scripts moved
to a bind mount (see common/Dockerfile.common) most commits cannot change image
content at all.

The tag is now a hash of exactly the repository content that `docker build`
bakes into the image, so an image is rebuilt or fetched if and only if it
actually differs. Identical content yields an identical tag on every node and
on every branch, which means `image_exist()` simply hits and no build, retag or
pull happens at all.

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
    """
    return [
        "common",
        "fingerprint_src",
        f"workloads/{workload_group}",
        ":(exclude)common/scripts",
        f":(exclude)workloads/{workload_group}/workload_root_entrypoint.sh",
        f":(exclude)workloads/{workload_group}/workload_user_entrypoint.sh",
    ]


def _tracked_files(workload_group: str, infra_dir) -> list:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--"] + image_context_pathspec(workload_group),
        cwd=str(infra_dir),
        check=True,
        capture_output=True,
    ).stdout
    return sorted(p for p in out.split(b"\0") if p)


_HASH_CACHE = {}


def image_content_hash(workload_group: str, infra_dir=None) -> str:
    """Hash the on-disk content of everything baked into the workload's image.

    Hashing the working tree rather than the git index means an uncommitted
    Dockerfile edit produces a different tag, so a local experiment cannot
    silently reuse an image built from different sources.

    Note this hashes the build *inputs*, not the resulting image bits. Steps
    that reach the network (the DynamoRIO clone, the pin download, `apt
    install`) are not pinned by content, so two builds of the same tag can
    still differ -- the same caveat the git-hash tag had.
    """
    infra_dir = Path(infra_dir) if infra_dir else REPO_ROOT
    key = (workload_group, str(infra_dir))
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]

    digest = hashlib.sha256()
    for rel in _tracked_files(workload_group, infra_dir):
        path = infra_dir / rel.decode()
        digest.update(rel)
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except (OSError, IsADirectoryError):
            # A tracked path missing from the worktree (e.g. sparse checkout)
            # is itself part of the identity; record it rather than ignoring it.
            digest.update(b"<absent>")
        digest.update(b"\0")

    result = digest.hexdigest()[:TAG_HASH_LEN]
    _HASH_CACHE[key] = result
    return result


def image_tag_for(workload_group: str, infra_dir=None) -> str:
    """Full local image reference, e.g. 'spec2017:3f9a12c0be44'."""
    return f"{workload_group}:{image_content_hash(workload_group, infra_dir)}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <workload_group>", file=sys.stderr)
        raise SystemExit(2)
    print(image_content_hash(sys.argv[1]))
