"""The version of annoray (Rust crates and Python package alike), derived from git tags.

Tags are ``vX.Y.Z`` for releases and ``vX.Y.Z-preN`` for pre-releases. At a tag the version is
the tag. M commits after one it is

- after ``vX.Y.Z``: ``X.Y.(Z+1)-dev.M+g<commit>``;
- after ``vX.Y.Z-preN``: ``X.Y.Z-pre(N+1).dev.M+g<commit>``;
- with no tag at all: ``0.0.1-dev.M+g<commit>`` (M counting every commit).

maturin turns these into Python versions (PEP 440 reads ``pre`` as ``rc``): ``0.3.0-pre1`` ->
``0.3.0rc1``, ``0.3.0-pre2.dev.2`` -> ``0.3.0rc2.dev2``, ``0.2.1-dev.7`` -> ``0.2.1.dev7``. Python
sorts a ``.dev`` build before the version it leads to; Cargo sorts ``0.3.0-pre2.dev.2`` after
``0.3.0-pre2``, which only matters if a dev build is ever published. (Cargo also compares
``pre9`` and ``pre10`` as text, so ``pre10`` would sort first: stop at ``pre9``.)

Cargo has no dynamic versions, so the version is stamped into the workspace ``Cargo.toml``
(``workspace.package.version`` and the internal ``workspace.dependencies``) before building
or publishing; the Python package takes it from there.

    python3 tools/version.py            print the version
    python3 tools/version.py --write    stamp it into Cargo.toml
    python3 tools/version.py --release  stamp it, failing unless HEAD is exactly at a tag
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARGO = ROOT / "Cargo.toml"
# Crates published from this workspace, whose `workspace.dependencies` entries carry a version.
INTERNAL = (
    "annoray",
    "annoray-codecs",
    "annoray-hdf5",
    "annoray-macros",
    "annoray-pyo3",
    "annoray-units",
    "annoray-zarr",
)
TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)(?:-pre(\d+))?")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, encoding="utf-8"
    ).stdout.strip()


def derive() -> tuple[str, bool]:
    """The version, and whether HEAD is exactly at a release tag."""
    try:
        described = git("describe", "--tags", "--long", "--match", "v[0-9]*")
    except subprocess.CalledProcessError:
        count, sha = (
            git("rev-list", "--count", "HEAD"),
            git("rev-parse", "--short", "HEAD"),
        )
        return f"0.0.1-dev.{count}+g{sha}", False
    tag, count, sha = described.rsplit("-", 2)
    m = TAG.fullmatch(tag)
    if m is None:
        sys.exit(f"tag {tag!r} is not vX.Y.Z or vX.Y.Z-preN")
    if count == "0":
        return tag.removeprefix("v"), True
    major, minor, patch, pre = m.groups()
    if pre is not None:
        return f"{major}.{minor}.{patch}-pre{int(pre) + 1}.dev.{count}+{sha}", False
    return f"{major}.{minor}.{int(patch) + 1}-dev.{count}+{sha}", False


def stamp(version: str) -> None:
    text = CARGO.read_text(encoding="utf-8")
    text, n = re.subn(
        r'(\[workspace\.package\][^\[]*?\nversion = )"[^"]*"',
        rf'\g<1>"{version}"',
        text,
    )
    if n != 1:
        sys.exit("no version in [workspace.package]")
    # Internal path dependencies need the exact version (build metadata is not part of it).
    plain = version.split("+", 1)[0]
    for name in INTERNAL:
        text, n = re.subn(
            rf'(\n{re.escape(name)} = \{{ path = "[^"]*", version = )"[^"]*"',
            rf'\g<1>"={plain}"',
            text,
        )
        if n != 1:
            sys.exit(f"no workspace dependency {name!r} with a version")
    CARGO.write_text(text, encoding="utf-8")


def main() -> None:
    version, at_tag = derive()
    if "--release" in sys.argv[1:] and not at_tag:
        sys.exit(f"not at a release tag (would be {version})")
    if {"--write", "--release"} & set(sys.argv[1:]):
        stamp(version)
    print(version)


if __name__ == "__main__":
    main()
