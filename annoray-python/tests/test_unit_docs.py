"""The unit tables of the documentation (``docs/user-guide/units.md``) against the units
engine: every symbol in a table parses and equals the value the table gives for it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import annoray as ar

PAGE = Path(__file__).resolve().parents[2] / "docs" / "user-guide" / "units.md"
ROW = re.compile(r"^\| ([^|]+) \| ([^|]+) \| ([^|]+) \|$")
CODE = re.compile(r"`([^`]+)`")


def table_rows() -> list[tuple[str, str, str]]:
    if not PAGE.is_file():
        return []
    rows = []
    for line in PAGE.read_text().splitlines():
        m = ROW.match(line.strip())
        if not m or m.group(1).strip() in ("Unit", "Form", "Written") or "---" in line:
            continue
        name, symbols, value = (g.strip() for g in m.groups())
        values = CODE.findall(value) or [value]
        syms = CODE.findall(symbols)
        # `min`, `h`, `d` | `60 s`, `3600 s`, `86400 s`: one value per symbol; else shared.
        paired = values if len(values) == len(syms) else [values[0]] * len(syms)
        rows.extend((name, sym, v) for sym, v in zip(syms, paired, strict=True))
    return rows


ROWS = table_rows()


@pytest.mark.skipif(not ROWS, reason="documentation sources not present")
@pytest.mark.parametrize(("name", "symbol", "value"), ROWS, ids=[f"{r[1]}" for r in ROWS])
def test_documented_unit(name: str, symbol: str, value: str) -> None:
    unit = ar.Unit(symbol)
    if value == "base":
        assert unit.si_factor == 1.0, name
        return
    # Equal as values: the same at 0 and at 1 (offset units included).
    for x in (0.0, 1.0):
        got = ar.Quantity(x, symbol).to(value).magnitude
        assert got == pytest.approx(x, rel=1e-12, abs=1e-9), f"{name}: 1 {symbol} != {value}"


@pytest.mark.skipif(not PAGE.is_file(), reason="documentation sources not present")
def test_symbols_to_read_carefully_parse() -> None:
    text = PAGE.read_text().split("### Symbols to read carefully", 1)[1].split("\n## ", 1)[0]
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if not m or "---" in line or m.group(1).strip() == "Written":
            continue
        for sym in CODE.findall(m.group(1)):
            ar.Unit(sym)
