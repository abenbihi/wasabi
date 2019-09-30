#!/bin/sh

# source useful path
. ./scripts/export_path.sh
img_dir="$WS_DIR"datasets/Extended-CMU-Seasons/

python3 -m tools.seg \
  --img_dir "$img_dir" \
  --slice_id 24 \
  --cam_id 0 \
  --survey_id -1
