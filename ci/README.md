# CI's Python packages

Every package CI installs from PyPI is locked to an exact version in `constraints.txt`, so a new
release of a dependency never changes a CI run on its own. The workflows install what each job
needs with `pip install ... -c ci/constraints.txt`: the job still installs only its own packages,
at the locked versions.

- `constraints.in` names what CI installs: annoray's `test` extra, `docs/requirements.txt`, and
  the tools and references CI adds beside them. The version ranges stay in
  `annoray-python/pyproject.toml` and `docs/requirements.txt`, so installing `annoray[test]`
  outside CI is not pinned.
- `constraints.txt` is generated from it (`python tools/lock_ci.py`; needs `uv`). The resolution
  is universal: valid on every Python from 3.11 and on Linux, macOS and Windows, with markers
  where a package needs different versions (numpy, scipy and zarr have dropped Python 3.11).

To move to newer versions:

```sh
pip install uv
python tools/lock_ci.py --upgrade
```

then run the test suite against the new pins (locally and in CI) before committing them. The
maturin that builds the release's wheels (`maturin-version` in `.github/workflows/release.yml`)
is kept equal to the `maturin` pin by hand.

Every job's packages have binary wheels at the locked versions, except `bottleneck` on Intel
macOS (no final release publishes those wheels), which pip builds from source there.
