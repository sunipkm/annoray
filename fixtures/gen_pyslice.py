"""Generate the reference table for `PySlice` normalisation.

Each output line is ``len start stop step|i,j,k`` where ``N`` stands for ``None`` and
``i,j,k`` are the indices Python's ``range(len)[start:stop:step]`` selects.

Run from the repository root:  python3 fixtures/gen_pyslice.py > fixtures/pyslice.txt
"""

BOUNDS = [None, *range(-6, 7)]
STEPS = [None, -3, -2, -1, 1, 2, 3]


def fmt(v):
    return "N" if v is None else str(v)


def main():
    for n in range(5):
        for start in BOUNDS:
            for stop in BOUNDS:
                for step in STEPS:
                    picked = list(range(n)[start:stop:step])
                    print(
                        f"{n} {fmt(start)} {fmt(stop)} {fmt(step)}|"
                        + ",".join(map(str, picked))
                    )


if __name__ == "__main__":
    main()
