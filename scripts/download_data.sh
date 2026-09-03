#!/usr/bin/env bash
# Download ODIR-5K from Kaggle into data/odir5k.
#
# Prerequisites:
#   1. pip install kaggle
#   2. kaggle.com -> your avatar -> Settings -> API -> Create New Token
#      This downloads kaggle.json.
#   3. Linux/macOS:  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
#                    chmod 600 ~/.kaggle/kaggle.json
#      Windows:      move kaggle.json to C:\Users\<you>\.kaggle\kaggle.json
#   4. Accept the dataset terms once by visiting the dataset page in a browser.
#
# The download is roughly 2 GB. Never commit it.

set -euo pipefail

SLUG="andrewmvd/ocular-disease-recognition-odir5k"
DEST="data/odir5k"

mkdir -p "$DEST"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Run: pip install kaggle" >&2
  exit 1
fi

echo "Downloading $SLUG into $DEST ..."
kaggle datasets download -d "$SLUG" -p "$DEST" --unzip

echo "Done. Expected contents:"
echo "  $DEST/full_df.csv"
echo "  $DEST/preprocessed_images/"
ls -la "$DEST" | head -20
