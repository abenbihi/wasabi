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

for slice_id in 22 23 24 25
do
  for cam_id in 0 1 
  do
    for survey_id in 0 1 2 3 4 5 6 7 8 9
    do
      if [ "$slice_id" -eq 24 ] || [ "$slice_id" -eq 25 ]; then
        if [ "$cam_id" -eq 1 ] && [ "$survey_id" -eq 8 ]; then
          echo "This traversal has no ground-truth pose."
          continue
        fi
      fi
      
      echo "\n\n** Slice "$slice_id" Cam "$cam_id" Survey "$survey_id" **"

      #python3 -m methods.wasabi.test_retrieve \
      #python3 -m tests.test_describe \
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
        --data cmu 

    if [ "$?" -ne 0 ]; then 
      echo "Error in run "$survey_id""
      exit 1
    fi

    done
  done
done
