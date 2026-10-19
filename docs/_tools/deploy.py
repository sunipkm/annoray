"""Place a documentation build in a checkout of the ``gh-pages`` branch.

Every release keeps its own directory (``0.3.1/``, ``0.4.0-pre1/``), which later releases
never touch, so the documentation of any published version stays reachable. Besides those:

- ``stable/`` is a copy of the newest final (not pre-) release, so links into it follow
  releases;
- ``dev/`` holds builds made by hand from a branch;
- ``switcher.json`` (the theme's version menu) is rebuilt from the directories present:
  ``dev``, then every release, newest first; the newest final release is preferred;
- the site root redirects to ``stable/`` (before the first final release, to the newest
  pre-release; before any release, to ``dev/``).

    python docs/_tools/deploy.py --site SITE --html docs/_build/html --version 0.3.1 --stable
    python docs/_tools/deploy.py --site SITE --html docs/_build/html --version 0.4.0-pre1
    python docs/_tools/deploy.py --site SITE --html docs/_build/html --version dev

Deploying a version that is already present replaces only that version (a rebuild of the
same release).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

RELEASE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-pre(\d+))?$")


def key(version: str) -> tuple[int, int, int, int, int]:
    """Sort key: a pre-release sorts before its final release."""
    m = RELEASE.match(version)
    if m is None:
        raise ValueError(version)
    major, minor, patch, pre = m.groups()
    final = pre is None
    return (int(major), int(minor), int(patch), int(final), int(pre or 0))


def is_final(version: str) -> bool:
    m = RELEASE.match(version)
    return m is not None and m.group(4) is None


def replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def releases(site: Path) -> list[str]:
    found = [p.name for p in site.iterdir() if p.is_dir() and RELEASE.match(p.name)]
    return sorted(found, key=key, reverse=True)


def switcher(site: Path, base_url: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if (site / "dev").is_dir():
        entries.append({"name": "dev", "version": "dev", "url": f"{base_url}/dev/"})
    found = releases(site)
    finals = [v for v in found if is_final(v)]
    preferred = finals[0] if finals else (found[0] if found else None)
    for v in found:
        entry: dict[str, object] = {"version": v, "url": f"{base_url}/{v}/"}
        if v == preferred and is_final(v):
            entry["name"] = f"{v} (stable)"
        elif not is_final(v):
            entry["name"] = f"{v} (pre-release)"
        else:
            entry["name"] = v
        if v == preferred:
            entry["preferred"] = True
        entries.append(entry)
    return entries


def redirect(target: str) -> str:
    return (
        '<!doctype html>\n<meta charset="utf-8">\n'
        f'<meta http-equiv="refresh" content="0; url={target}">\n'
        f'<link rel="canonical" href="{target}">\n'
        f'<a href="{target}">annoray documentation</a>\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument(
        "--site", type=Path, required=True, help="checkout of the gh-pages branch"
    )
    ap.add_argument("--html", type=Path, required=True, help="the Sphinx HTML build")
    ap.add_argument(
        "--version", required=True, help="'dev' or a release: X.Y.Z or X.Y.Z-preN"
    )
    ap.add_argument(
        "--stable",
        action="store_true",
        help="also publish as stable/ (when this is the newest final release)",
    )
    ap.add_argument("--base-url", default="https://sunipkm.github.io/annoray")
    a = ap.parse_args()
    if a.version != "dev" and not RELEASE.match(a.version):
        raise SystemExit(
            f"version must be 'dev', X.Y.Z or X.Y.Z-preN, not {a.version!r}"
        )
    if not (a.html / "index.html").is_file():
        raise SystemExit(f"{a.html} is not an HTML build")
    base = a.base_url.rstrip("/")

    replace_dir(a.html, a.site / a.version)
    finals = [v for v in releases(a.site) if is_final(v)]
    if a.stable and finals and finals[0] == a.version:
        replace_dir(a.html, a.site / "stable")
    (a.site / "switcher.json").write_text(
        json.dumps(switcher(a.site, base), indent=2) + "\n"
    )
    if (a.site / "stable").is_dir():
        target = "stable/"
    elif releases(a.site):
        target = f"{releases(a.site)[0]}/"
    else:
        target = "dev/"
    (a.site / "index.html").write_text(redirect(target))
    (a.site / ".nojekyll").write_text("")
    print(f"deployed {a.version} to {a.site} (root -> {target})")


if __name__ == "__main__":
    main()
