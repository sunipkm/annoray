#!/usr/bin/env bash
# Build the documentation locally with the checks CI runs, and optionally serve it.
#
#   docs/build.sh                  build (incremental) into docs/_build/html
#   docs/build.sh --clean          rebuild every page (runs every example again)
#   docs/build.sh --serve [PORT]   build, then serve on http://localhost:PORT (default 8000)
#   docs/build.sh --skip-install   use the installed extension and packages as they are
#
# Steps: build and install the extension (maturin), install docs/requirements.txt, check that
# every public name is in docs/api/*.md, then sphinx-build with warnings as errors.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

clean=0
serve=0
install=1
port=8000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) clean=1 ;;
        --serve)
            serve=1
            if [[ $# -gt 1 && "$2" =~ ^[0-9]+$ ]]; then
                port="$2"
                shift
            fi
            ;;
        --skip-install) install=0 ;;
        -h | --help)
            sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: $1 (see --help)" >&2
            exit 2
            ;;
    esac
    shift
done

python="${PYTHON:-python3}"

if [[ $install -eq 1 ]]; then
    echo "==> Building the extension (maturin develop --release)"
    (cd annoray-python && maturin develop --release)
    echo "==> Installing documentation requirements"
    "$python" -m pip install --quiet -r docs/requirements.txt
fi

echo "==> Checking the API reference covers every public name"
"$python" docs/_tools/check_api.py

if [[ $clean -eq 1 ]]; then
    rm -rf docs/_build
fi

echo "==> Building HTML (warnings are errors)"
"$python" -m sphinx -b html -W --keep-going -d docs/_build/doctrees docs docs/_build/html

echo "==> Built: docs/_build/html/index.html"
if [[ $serve -eq 1 ]]; then
    echo "==> Serving on http://localhost:$port (Ctrl-C to stop)"
    exec "$python" -m http.server --directory docs/_build/html "$port"
fi
