# Documentation sources

The documentation site (Sphinx, pydata-sphinx-theme, MyST): user pages in Markdown, the
Python API reference generated from docstrings, with signatures from the type stubs
(`_ext/stubsig.py`). `.github/workflows/docs.yml` builds and checks it; a release waits for
that build as for the CI, and `docs-deploy.yml` publishes it to GitHub Pages once the
release is out: each release in its own directory (`/0.3.1/`, `/0.4.0-pre1/`), never replaced
by later releases, and the newest final release also as `/stable/`. `docs-dev.yml`, run by
hand, publishes `/dev/`.

`docs/build.sh` runs the same steps as CI: it builds and installs the extension, installs
`requirements.txt`, checks the API reference, and builds with warnings as errors.

```sh
docs/build.sh                  # build into docs/_build/html
docs/build.sh --serve          # build, then serve on http://localhost:8000
docs/build.sh --clean --serve  # rebuild every page (every example runs again)
docs/build.sh --skip-install   # use the installed extension and packages as they are
```

The steps by hand:

```sh
cd annoray-python && maturin develop --release && cd ..
pip install -r docs/requirements.txt
python docs/_tools/check_api.py                 # every public name is in docs/api/*.md
sphinx-build -b html -W --keep-going -d docs/_build/doctrees docs docs/_build/html
```

- `api/*.md`: the reference pages. A new public method or function goes into the right
  section there; `check_api.py` fails until it does.
- `getting-started/overview.md` is executed at build time (myst-nb): its output is real.
- `_tools/deploy.py`: places a build in a checkout of `gh-pages` and rebuilds the version
  switcher (`switcher.json`).
