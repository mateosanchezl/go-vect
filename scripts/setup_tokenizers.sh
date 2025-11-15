#!/usr/bin/env bash
set -euo pipefail

LIB_DIR="libs/tokenizers"
LIB_FILE="${LIB_DIR}/libtokenizers.a"
TOKENIZERS_REPO="https://github.com/daulet/tokenizers.git"
TOKENIZERS_COMMIT="e9a73faf2b69c8d04d56decd978c073d55311c16"

log() {
  printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$1"
}

fail() {
  printf "ERROR: %s\n" "$1" >&2
  exit 1
}

check_deps() {
  for dep in cargo git; do
    if ! command -v "$dep" >/dev/null 2>&1; then
      fail "Missing dependency: $dep. Please install Rust toolchain from https://rustup.rs/ and git"
    fi
  done
}

ensure_correct_commit() {
  # Create parent directory if needed
  mkdir -p "$(dirname "$LIB_DIR")"
  
  if [ ! -d "${LIB_DIR}/.git" ]; then
    # Directory doesn't exist or isn't a git repo, clone it
    log "Tokenizers source not found. Cloning repository..."
    
    # Clone into a temp directory first to avoid issues if libs/tokenizers exists but is empty
    TEMP_DIR=$(mktemp -d)
    if ! git clone "$TOKENIZERS_REPO" "$TEMP_DIR/tokenizers"; then
      rm -rf "$TEMP_DIR"
      fail "Failed to clone tokenizers repository"
    fi
    
    # Move to final location
    if [ -d "$LIB_DIR" ] && [ "$(ls -A "$LIB_DIR" 2>/dev/null)" ]; then
      log "Warning: ${LIB_DIR} exists and is not empty. Removing and re-cloning..."
      rm -rf "$LIB_DIR"
    fi
    
    mv "$TEMP_DIR/tokenizers" "$LIB_DIR"
    rm -rf "$TEMP_DIR"
    log "✓ Tokenizers repository cloned successfully"
  fi
  
  # Ensure we're on the correct commit
  cd "$LIB_DIR"
  CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
  
  if [ "$CURRENT_COMMIT" != "$TOKENIZERS_COMMIT" ]; then
    log "Checking out commit ${TOKENIZERS_COMMIT}..."
    # Fetch all refs from origin to ensure we have the commit
    log "Fetching latest changes from origin..."
    if ! git fetch origin; then
      fail "Failed to fetch from origin"
    fi
    
    if ! git checkout "$TOKENIZERS_COMMIT" 2>/dev/null; then
      fail "Failed to checkout commit ${TOKENIZERS_COMMIT}. It may not exist in the repository."
    fi
    log "✓ Checked out commit ${TOKENIZERS_COMMIT}"
  else
    log "Already on correct commit ${TOKENIZERS_COMMIT}"
  fi
  
  cd - > /dev/null
}

main() {
  log "Checking dependencies..."
  check_deps

  log "Creating libs directory at ${LIB_DIR}..."
  mkdir -p "$LIB_DIR"

  # Ensure tokenizers is cloned and on the correct commit
  ensure_correct_commit

  log "Building libtokenizers..."
  cd "$LIB_DIR"
  
  if [ ! -f "Cargo.toml" ]; then
    fail "Cargo.toml not found in ${LIB_DIR} after clone attempt."
  fi

  log "Running cargo build --release..."
  if ! cargo build --release; then
    fail "Failed to build libtokenizers"
  fi

  log "Copying libtokenizers.a to ${LIB_DIR}..."
  if [ -f "target/release/libtokenizers.a" ]; then
    cp target/release/libtokenizers.a .
    log "✓ libtokenizers.a copied successfully"
  else
    fail "Built library not found at target/release/libtokenizers.a"
  fi

  cd - > /dev/null

  if [ ! -f "$LIB_FILE" ]; then
    fail "Library file not found at ${LIB_FILE} after build"
  fi

  log "Tokenizers library built successfully at: ${LIB_FILE}"
}

main "$@"

