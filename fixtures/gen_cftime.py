"""Generate the reference table for CF time decoding, using cftime.

Each output line is ``calendar|units|value|result`` where ``result`` is
``year month day hour minute second microsecond`` or ``ERR``. Integer values are written
without a decimal point; floats are written with ``repr`` so they round-trip exactly.

Run from the repository root:  python3 fixtures/gen_cftime.py > fixtures/cftime.txt
"""

import warnings

import cftime

warnings.simplefilter("ignore")

CALENDARS = ["standard", "proleptic_gregorian", "julian", "noleap", "all_leap", "360_day"]
REFS = [
    "1970-01-01",
    "1582-10-15 00:00:00",
    "1582-10-04",
    "0001-01-01",
    "1850-01-01 12:30:15",
    "2000-03-01T06:00:00Z",
    "2000-01-01 00:00:00 +05:30",
]
STEPS = ["days", "hours", "minutes", "seconds", "milliseconds", "microseconds"]
INTS = [-1_000_000, -146_097, -36_525, -1, 0, 1, 59, 365, 366, 1461, 146_097, 730_120, 10_000_000]
FLOATS = [0.1, 1.5, -0.25, 1 / 3, 0.123456789, 100000.5, -12345.678]


def main():
    print(f"# cftime {cftime.__version__}")
    for cal in CALENDARS:
        for ref in REFS:
            for step in STEPS:
                units = f"{step} since {ref}"
                for v in INTS + FLOATS:
                    try:
                        d = cftime.num2date(v, units, calendar=cal)
                        res = f"{d.year} {d.month} {d.day} {d.hour} {d.minute} {d.second} {d.microsecond}"
                    except Exception:
                        res = "ERR"
                    print(f"{cal}|{units}|{v!r}|{res}")


if __name__ == "__main__":
    main()
