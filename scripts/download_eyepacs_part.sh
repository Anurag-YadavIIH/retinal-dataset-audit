#!/bin/bash
# Download one EyePACS train.zip part with real stall detection: kills and
# retries only if the file's own byte size stops growing for STALL_SECS,
# not a blind time cap (a flat timeout killed every attempt on the first
# try because 8GB at ~11-12MB/s genuinely needs ~12 minutes -- shorter than
# the 300s cap that produced that failure).
set -u
PART="$1"
DEST_DIR="/d/retinaprep_data/eyepacs"
DEST_FILE="$DEST_DIR/train.zip.$PART.zip"
STALL_SECS=120
POLL_SECS=15
MAX_ATTEMPTS=3

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== train.zip.$PART attempt $attempt ==="
  rm -f "$DEST_FILE" "$DEST_FILE.kaggle-partial"

  ./.venv/Scripts/python -m kaggle competitions download -c diabetic-retinopathy-detection \
    -f "train.zip.$PART" -p "$DEST_DIR" --force &
  DL_PID=$!

  last_size=-1
  stalled_for=0
  while kill -0 "$DL_PID" 2>/dev/null; do
    sleep "$POLL_SECS"
    cur_size=$(stat -c%s "$DEST_FILE" 2>/dev/null || echo 0)
    if [ "$cur_size" -eq "$last_size" ]; then
      stalled_for=$((stalled_for + POLL_SECS))
    else
      stalled_for=0
      last_size=$cur_size
    fi
    if [ "$stalled_for" -ge "$STALL_SECS" ]; then
      echo "=== train.zip.$PART stalled at ${cur_size} bytes for ${stalled_for}s -- killing ==="
      kill -9 "$DL_PID" 2>/dev/null
      wait "$DL_PID" 2>/dev/null
      break
    fi
  done
  wait "$DL_PID" 2>/dev/null
  status=$?

  if [ "$status" -eq 0 ] && [ -f "$DEST_FILE" ]; then
    echo "=== train.zip.$PART OK ($(stat -c%s "$DEST_FILE") bytes) ==="
    exit 0
  fi
  echo "=== train.zip.$PART attempt $attempt did not complete cleanly (status=$status), retrying ==="
done

echo "=== train.zip.$PART FAILED after $MAX_ATTEMPTS attempts ==="
exit 1
