"""Lock CI's Python packages: resolve ``ci/constraints.in`` into exact versions in
``ci/constraints.txt``, valid on every Python from 3.11 and on Linux, macOS and Windows (``uv pip
compile --universal``; packages that need different versions get environment markers).

    python tools/lock_ci.py              resolve, keeping the current pins where they still fit
    python tools/lock_ci.py --upgrade    resolve every package to its newest version

Needs ``uv`` (``pip install uv``). After upgrading, run the test suite against the new pins
before committing them (see ``ci/README.md``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / "ci"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--upgrade", action="store_true", help="newest versions of everything")
    a = ap.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv not found: pip install uv")
    cmd = [
        uv,
        "pip",
        "compile",
        "constraints.in",
        "--universal",
        "--python-version",
        "3.11",
        # The package under test is always installed from the build being tested.
        "--no-emit-package",
        "annoray",
        "--custom-compile-command",
        "python tools/lock_ci.py",
        "--output-file",
        "constraints.txt",
    ]
    if a.upgrade:
        cmd.append("--upgrade")
    subprocess.run(cmd, cwd=CI, check=True)


if __name__ == "__main__":
    main()
