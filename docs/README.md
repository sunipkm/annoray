# Documentation sources

The documentation site (Sphinx, pydata-sphinx-theme, MyST): user pages in Markdown, the
Python API reference generated from docstrings, with signatures from the type stubs
(`_ext/stubsig.py`). Published to GitHub Pages by `.github/workflows/docs.yml`: each release
into `/X.Y/` and `/stable/`, builds by hand into `/dev/`.

```sh
cd annoray-python && maturin develop --release && cd ..
pip install -r docs/requirements.txt
python docs/_tools/check_api.py                 # every public name is in docs/api/*.md
sphinx-build -b html -W -d docs/_build/doctrees docs docs/_build/html
```

- `api/*.md`: the reference pages. A new public method or function goes into the right
  section there; `check_api.py` fails until it does.
- `getting-started/overview.md` is executed at build time (myst-nb): its output is real.
- `_tools/deploy.py`: places a build in a checkout of `gh-pages` and rebuilds the version
  switcher (`switcher.json`).
