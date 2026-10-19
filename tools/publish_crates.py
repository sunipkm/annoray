"""Publish the workspace crates to crates.io, safely re-runnable.

Crates go in dependency order (from ``cargo metadata``). A crate whose version is already on
crates.io is skipped, so a release whose publishing stopped part-way is finished by running
this again. When crates.io refuses with 429 Too Many Requests (it limits how many *new*
crates an account publishes in a burst), the script waits until the time crates.io names and
tries again.

    python3 tools/publish_crates.py            # needs CARGO_REGISTRY_TOKEN
    python3 tools/publish_crates.py --dry-run  # list what would be published

The version must already be stamped into Cargo.toml (``tools/version.py --release``).
"""

from __future__ import annotations

import argparse
import email.utils
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "annoray-release (https://github.com/sunipkm/annoray)"
# Longest single wait for a rate limit before giving up (crates.io's new-crate window is
# minutes, not hours).
MAX_WAIT_S = 3600
TRIES = 5


def workspace_order() -> list[tuple[str, str]]:
    """Publishable workspace crates as (name, version), dependencies first."""
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    members = {
        p["name"]: p
        for p in meta["packages"]
        if p["id"] in meta["workspace_members"] and p.get("publish") != []
    }
    order: list[str] = []

    def visit(name: str, seen: frozenset[str] = frozenset()) -> None:
        if name in order:
            return
        if name in seen:
            raise SystemExit(f"dependency cycle through {name}")
        for dep in members[name]["dependencies"]:
            if dep["name"] in members and dep.get("kind") != "dev":
                visit(dep["name"], seen | {name})
        order.append(name)

    for name in sorted(members):
        visit(name)
    return [(n, members[n]["version"]) for n in order]


def published(name: str, version: str) -> bool:
    url = f"https://crates.io/api/v1/crates/{name}/{version}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def retry_after(output: str) -> float | None:
    """Seconds until crates.io accepts new crates again, from a 429 message."""
    if "429" not in output:
        return None
    m = re.search(r"try again after (.+? GMT)", output)
    if not m:
        return 600.0
    when = email.utils.parsedate_to_datetime(m.group(1))
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds()) + 15.0


def publish(name: str) -> None:
    for attempt in range(1, TRIES + 1):
        r = subprocess.run(
            ["cargo", "publish", "-p", name, "--allow-dirty"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode == 0:
            return
        wait = retry_after(r.stderr + r.stdout)
        if wait is None or wait > MAX_WAIT_S or attempt == TRIES:
            raise SystemExit(f"publishing {name} failed")
        print(
            f"crates.io rate limit: waiting {wait:.0f} s before retrying {name}",
            flush=True,
        )
        time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument(
        "--dry-run", action="store_true", help="only list what would be published"
    )
    a = ap.parse_args()
    for name, version in workspace_order():
        if published(name, version):
            print(f"{name} {version}: already on crates.io, skipped", flush=True)
            continue
        if a.dry_run:
            print(f"{name} {version}: would publish", flush=True)
            continue
        print(f"{name} {version}: publishing", flush=True)
        publish(name)


if __name__ == "__main__":
    main()
