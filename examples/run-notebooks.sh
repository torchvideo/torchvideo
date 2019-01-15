#!/usr/bin/env bash
NOTEBOOK_DIR="$(dirname $(readlink -f $0))"
set -ex

for f in "$NOTEBOOK_DIR"/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$f"
done
