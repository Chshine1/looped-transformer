#!/usr/bin/env bash
set -euo pipefail

CLUTRR_DIR=".data-sources/clutrr-v1.3"
CLUTRR_URL="org-16943930@github.com:facebookresearch/clutrr.git"
COMMIT="4f65217"

if [ -d "$CLUTRR_DIR" ] && [ -d "$CLUTRR_DIR/.git" ]; then
  echo "CLUTRR already cloned, skipping clone."
else
  echo "Cloning CLUTRR into $CLUTRR_DIR ..."
  rm -rf "$CLUTRR_DIR"
  mkdir -p ".data-sources"
  git clone --recurse-submodules "$CLUTRR_URL" "$CLUTRR_DIR"
fi

cd "$CLUTRR_DIR"
git fetch origin
git checkout "$COMMIT"
git submodule update --init --recursive
echo "CLUTRR is ready at $(pwd) (commit $COMMIT)."