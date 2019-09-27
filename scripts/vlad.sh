#!/bin/sh

MACHINE=1
if [ "$MACHINE" -eq 0 ]; then
  ws_dir=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  ws_dir=/home/gpu_user/assia/ws/
else
  echo "Get your MTF MACHINE macro correct. Bye !"
  exit 1
fi


if [ $# -eq 0 ]; then
  echo "Usage"
  echo "1. trial "
  echo "2. max_num_feat"
  exit 0
fi

if [ $# -ne 2 ]; then
  echo "Bad number of arguments"
  echo "1. trial "
  echo "2. max_num_feat"
  exit 1
fi

# source useful path
. ./scripts/export_path.sh


trial="$1"
max_num_feat="$2"
n_words=64
top_k=20

out_dir=res/vlad/"$trial"
#if [ -d "$out_dir" ]; then
#    while true; do
#        read -p ""$out_dir" already exists. Do you want to overwrite it (y/n) ?" yn
#        case $yn in
#            [Yy]* ) rm -rf "$out_dir"; mkdir -p "$out_dir"; break;;
#            [Nn]* ) exit;;
#            * ) * echo "Please answer yes or no.";;
#        esac
#    done
#else
#    mkdir -p "$out_dir"
#fi

img_dir="$WS_DIR"datasets/Extended-CMU-Seasons/
meta_dir=meta/cmu/surveys/
seg_dir="$WS_DIR"tf/cross-season-segmentation/res/ext_cmu/

for slice_id in 22 # 23 24 25
do
  for cam_id in 0 #1 
  do
    for survey_id in 0 #1 2 3 4 5 6 7 8 9
    do
      if [ "$slice_id" -eq 24 ] || [ "$slice_id" -eq 25 ]; then
        if [ "$cam_id" -eq 1 ] && [ "$survey_id" -eq 8 ]; then
          echo "This traversal has no ground-truth pose."
          continue
        fi
      fi

      echo "\n\n** Slice "$slice_id" Cam "$cam_id" Survey "$survey_id" **"

      python3 -m methods.vlad.retrieve \
        --trial "$trial" \
        --dist_pos 5 \
        --top_k "$top_k" \
        --lf_mode sift \
        --max_num_feat "$max_num_feat" \
        --agg_mode vlad \
        --centroids flickr100k \
        --vlad_norm ssr \
        --data cmu \
        --slice_id "$slice_id" \
        --cam_id "$cam_id" \
        --survey_id "$survey_id" \
        --img_dir "$img_dir" \
        --meta_dir "$meta_dir" \
        --n_words "$n_words" \
        --resize 0 \
        --w 640 \
        --h 480 
    done
  done
done


