#!/usr/bin/env bash
# Build the documentation site. Usage: ./build_docs.sh [output dir]   (default: ../website/docs next to the clone)
set -e
cd "$(dirname "$0")"
cp DATASETS.md docs/guides/datasets.md                      # single source for the data page
OUT=${1:-../website/docs}
mkdocs build --clean -d "$OUT"
echo "built into $OUT"
