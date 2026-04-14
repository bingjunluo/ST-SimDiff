#!/bin/bash
# Prepare VideoMME dataset from downloaded chunk
# Run after videos_chunked_01.zip is fully downloaded

set -e

CHUNK_DIR="/tmp/videomme_chunks"
HF_HOME="${HOME}/.cache/huggingface"
VIDEOMME_DIR="${HF_HOME}/videomme/data"

mkdir -p "${VIDEOMME_DIR}"

echo "Extracting VideoMME videos from chunk1..."
if [ -f "${CHUNK_DIR}/videos_chunked_01.zip" ]; then
    # Check if fully downloaded (expected ~5.18GB)
    SIZE=$(stat -c %s "${CHUNK_DIR}/videos_chunked_01.zip" 2>/dev/null || echo 0)
    EXPECTED=5179377199
    echo "Chunk size: ${SIZE} bytes (expected: ~${EXPECTED})"

    # Extract
    unzip -n "${CHUNK_DIR}/videos_chunked_01.zip" -d "${VIDEOMME_DIR}/" 2>&1 | tail -5
    echo "Extraction complete"
else
    echo "ERROR: videos_chunked_01.zip not found at ${CHUNK_DIR}"
    exit 1
fi

# Count extracted videos
VIDEO_COUNT=$(ls "${VIDEOMME_DIR}/"*.mp4 2>/dev/null | wc -l)
echo "Videos available: ${VIDEO_COUNT}"

# List first few videos
echo "First 5 videos:"
ls "${VIDEOMME_DIR}/"*.mp4 2>/dev/null | head -5
