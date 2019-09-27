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
  echo "1. trial"
  echo "2. retrieval instance {cmu_park, cmu_urban, lake}"
  echo "3. aggregation mode"
  echo "4. centroids"
  exit 0
fi

if [ $# -ne 4 ]; then
  echo "Bad number of arguments"
  echo "1. trial"
  echo "2. retrieval instance {cmu_park, cmu_urban, lake}"
  echo "3. aggregation mode"
  echo "4. centroids"
  exit 1
fi

# source useful path
. ./scripts/export_path.sh


trial="$1"
instance="$2"
agg_mode="$3"
centroids="$4"

max_num_feat=1000
n_words=64
top_k=20

out_dir=res/vlad/"$trial"
if [ -d "$out_dir" ]; then
    while true; do
        read -p ""$out_dir" already exists. Do you want to overwrite it (y/n) ?" yn
        case $yn in
            [Yy]* ) rm -rf "$out_dir"; 
              mkdir -p "$out_dir"/retrieval; 
              mkdir -p "$out_dir"/perf; 
              break;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
    mkdir -p "$out_dir"/retrieval
    mkdir -p "$out_dir"/perf
fi

img_dir="$WS_DIR"datasets/Extended-CMU-Seasons/
meta_dir=meta/cmu/surveys/
seg_dir="$WS_DIR"tf/cross-season-segmentation/res/ext_cmu/

while read -r line
do
  slice_id=$(echo "$line" | cut -d' ' -f1)
  cam_id=$(echo "$line" | cut -d' ' -f2)
  for survey_id in $(echo "$line" | cut -d' ' -f 3-)
  do
      echo "\n\n** Slice "$slice_id" Cam "$cam_id" Survey "$survey_id" **"
      python3 -m methods.vlad_bow.retrieve \
        --trial "$trial" \
        --dist_pos 5 \
        --top_k "$top_k" \
        --lf_mode sift \
        --max_num_feat "$max_num_feat" \
        --agg_mode bow \
        --centroids "$centroids" \
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
      
      if [ "$?" -ne 0 ]; then
        echo "Error in slice "$slice_id" cam "$cam_id" survey_id "$survey_id""
        exit 1
      fi
    done
done < "$instance"


