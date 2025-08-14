#!/usr/bin/env bash
set -euo pipefail

S3_URL="https://storage.yandexcloud.net/dl-cv-home-test/instereo2k_sample.zip"
ZIP_NAME="instereo2k_sample.zip"
TARGET_DIR="data"

echo "📥 Downloading dataset (curl)…"
curl -L --fail --progress-bar "$S3_URL" -o "$ZIP_NAME"

echo "📦 Unzipping into '$TARGET_DIR/'…"
mkdir -p "$TARGET_DIR"
unzip -q "$ZIP_NAME" -d "$TARGET_DIR"

echo "🧹 Cleaning up…"
rm -f "$ZIP_NAME"

echo "✅ Done: extracted to $TARGET_DIR/"