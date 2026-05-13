#!/usr/bin/env bash
# backup_results.sh — snapshot data/results to a timestamped tar.gz
#
# Usage:
#   bash scripts/backup_results.sh                    # saves to ~/conjlean_results_<timestamp>.tar.gz
#   bash scripts/backup_results.sh /mnt/backup        # saves to specified directory
#   bash scripts/backup_results.sh . --upload         # also push to GitHub as a release asset
#
# The backup is placed OUTSIDE the repo directory so it survives a git clean.
# Run this before stopping a Lambda instance.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${1:-$HOME}"
UPLOAD="${2:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${DEST_DIR}/conjlean_results_${TIMESTAMP}.tar.gz"

echo "[backup] Source : ${REPO_ROOT}/data/results"
echo "[backup] Archive: ${ARCHIVE}"

if [ ! -d "${REPO_ROOT}/data/results" ]; then
    echo "[backup] ERROR: data/results not found. Nothing to back up."
    exit 1
fi

tar -czf "${ARCHIVE}" \
    -C "${REPO_ROOT}" \
    data/results

SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
echo "[backup] Done — ${SIZE} written to ${ARCHIVE}"

if [ "${UPLOAD}" = "--upload" ]; then
    if ! command -v gh &>/dev/null; then
        echo "[backup] gh CLI not found — skipping upload. Copy the archive manually."
        exit 0
    fi
    TAG="results-${TIMESTAMP}"
    echo "[backup] Creating GitHub release ${TAG} and uploading archive..."
    gh release create "${TAG}" "${ARCHIVE}" \
        --title "Experiment results ${TIMESTAMP}" \
        --notes "Automated backup of data/results from Lambda instance." \
        --prerelease
    echo "[backup] Uploaded to GitHub release ${TAG}"
fi
