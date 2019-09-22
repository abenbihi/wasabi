#!/bin/sh

if [ "$#" -eq 0 ]; then 
  echo "1. trial"
  exit 0
fi

if [ "$#" -ne 1 ]; then 
  echo "Error: bad number of arguments"
  echo "1. trial"
  exit 0
fi

trial="$1"

res_dir=res/wasabi/"$trial"

# source useful path
. ./scripts/export_path.sh

img_dir="$WS_DIR"datasets/Extended-CMU-Seasons/
meta_dir=meta/cmu/surveys/
seg_dir="$WS_DIR"tf/cross-season-segmentation/res/ext_cmu/

min_blob_size=50
min_contour_size=50
min_edge_size=40
contour_sample_num=64

slice_id=24
cam_id=0
survey_id=0

python3 -m methods.wasabi.retrieve \
  --trial "$trial" \
  --top_k 20 \
  --img_dir "$img_dir" \
  --seg_dir "$seg_dir" \
  --meta_dir "$meta_dir" \
  --dist_pos 5. \
  --min_blob_size "$min_blob_size" \
  --min_contour_size "$min_contour_size" \
  --min_edge_size "$min_edge_size" \
  --contour_sample_num "$contour_sample_num" \
  --slice_id "$slice_id" \
  --cam_id "$cam_id" \
  --survey_id "$survey_id" \
  --data cmu \
