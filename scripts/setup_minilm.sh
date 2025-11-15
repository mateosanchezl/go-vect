#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="sentence-transformers/all-MiniLM-L6-v2"
BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/main"
OUT_DIR="models/all-MiniLM-L6-v2"

FILES=(
  "onnx/model.onnx"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.txt"
  "config.json"
)

log() {
  printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$1"
}

fail() {
  printf "ERROR: %s\n" "$1" >&2
  exit 1
}

check_deps() {
  for dep in curl mkdir; do
    command -v "$dep" >/dev/null 2>&1 || fail "Missing dependency: $dep"
  done
}

download_file() {
  local file="$1"
  local url="${BASE_URL}/${file}"
  local dest="${OUT_DIR}/${file}"

  mkdir -p "$(dirname "$dest")"

  log "Downloading ${file}..."
  if ! curl -fsSL "$url" -o "$dest"; then
    fail "Failed to download ${file}"
  fi
}

main() {
  log "Checking dependencies..."
  check_deps

  log "Creating model directory at ${OUT_DIR}..."
  mkdir -p "$OUT_DIR"

  for f in "${FILES[@]}"; do
    download_file "$f"
  done

  log "All files downloaded successfully."
  log "Model ready at: ${OUT_DIR}"
}

main "$@"
