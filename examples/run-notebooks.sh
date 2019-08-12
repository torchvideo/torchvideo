#!/usr/bin/env bash
NOTEBOOK_DIR="$(dirname $(readlink -f $0))"
set -ex

for f in "$NOTEBOOK_DIR"/*.ipynb; do
    # Some cells can take a long time to execute, by default jupyter
    # only has a 30s timeout, so we increase this
    jupyter nbconvert --ExecutePreprocessor.timeout=120 --to notebook --execute --inplace "$f"
done
