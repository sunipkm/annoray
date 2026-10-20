"""Fail when a public name is missing from the API reference (``docs/api/*.md``).

Every class in ``annoray.__all__`` with its public members, and every public function, must
appear in an ``autosummary`` list. Run from anywhere: ``python docs/_tools/check_api.py``.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import annoray

API = Path(__file__).resolve().parent.parent / "api"
# Names documented elsewhere, or not API: options context manager internals and modules.
SKIP = {"__version__", "dims", "testing", "UnitError"}


def listed() -> set[str]:
    names: set[str] = set()
    for page in API.glob("*.md"):
        module = "annoray"
        for line in page.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\.\. currentmodule:: (\S+)", line.strip())
            if m:
                module = m.group(1)
                continue
            m = re.match(r"^ {3}([A-Za-z_][\w.]*)$", line)
            if m:
                name = m.group(1)
                full = f"{module}.{name}"
                names.add(full.removeprefix("annoray."))
    return names


def public() -> set[str]:
    want: set[str] = set()
    for name in annoray.__all__:
        if name in SKIP:
            continue
        obj = getattr(annoray, name)
        want.add(name)
        if inspect.isclass(obj) and not issubclass(obj, BaseException):
            want.update(
                f"{name}.{k}"
                for k in vars(obj)
                if not k.startswith("_") and name != "set_options"
            )
    return want


def main() -> int:
    missing = sorted(public() - listed())
    if missing:
        print("missing from docs/api/*.md:", *missing, sep="\n  ")
        return 1
    print(f"API reference covers all {len(public())} public names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
