#!/usr/bin/env bash
NOTEBOOK_DIR="$(dirname $(readlink -f $0))"
set -ex

for f in "$NOTEBOOK_DIR"/*.ipynb; do
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$f"
done
