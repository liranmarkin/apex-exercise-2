#!/usr/bin/env bash
# Download data files from GCS public bucket
# Usage: ./scripts/download_data.sh

set -euo pipefail

BUCKET_URL="https://storage.googleapis.com/apex-exercise-2-data"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

FILES=(
  "dataset/dataset-output/dataset/topic-information-blogs-manual-html-parse.json"
  "dataset/dataset-output/dataset/topics-faq.json"
  "dataset/dataset-output/dataset/topics-pdf-docling.json"
  "dataset/dataset-output/dataset/topics-policies-manual-html-parse.json"
  "dataset/dataset-output/rag_dumps/blogs_dump.json"
  "dataset/dataset-output/rag_dumps/policies_dump.json"
  "dataset/dataset-output/test-dataset/magen-mashlim-child.html/הראל-מגן-משלים-לילד-144.json"
  "dataset/dataset-output/test-dataset/magen-mashlim-child.html/הראל-מגן-משלים-לילד-144.pdf"
  "dataset/dataset-output/test-dataset/magen-mashlim-child.html/מידע-מהותי-הראל-מגן-משלים-לילד.json"
  "dataset/dataset-output/test-dataset/magen-mashlim-child.html/מידע-מהותי-הראל-מגן-משלים-לילד.pdf"
  "dataset/dataset-parse-1.zip"
  "dataset/dataset-raw.zip"
  "db/harel.db"
)

echo "Downloading data files from GCS..."

for file in "${FILES[@]}"; do
  dest="$ROOT_DIR/$file"
  mkdir -p "$(dirname "$dest")"

  if [ -f "$dest" ] && [ "$(wc -c < "$dest")" -gt 200 ]; then
    echo "  SKIP (exists): $file"
    continue
  fi

  echo "  Downloading: $file"
  curl -fSL --create-dirs -o "$dest" "$BUCKET_URL/$(python3 -c "import urllib.parse; print(urllib.parse.quote('$file', safe='/'))")"
done

echo "Done! All data files downloaded."
